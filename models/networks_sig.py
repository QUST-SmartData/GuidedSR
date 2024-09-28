import torch
import torch.nn as nn
from torch.nn import init
from jacobian import JacobianReg
import functools
from torch.optim import lr_scheduler
import numpy as np


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':#批量归一化
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        #卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定。
    elif norm_type == 'instance':#实例归一化
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler  返回学习速率调度程序

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], ds_fac=1):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer  最后一个conv层中的过滤器数
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'sr_resnet_9blocks':
        net = SRResnetGenerator(input_nc, output_nc, ds_fac, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'sr_resnet_6blocks':
        net = SRResnetGenerator(input_nc, output_nc, ds_fac, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'linearfilt':
        net = LinearModel(nonlinear=False)
    elif netG == 'gammafilt':
        net = LinearModel(nonlinear=True)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

# def cal_image_grad_penalty(image_out, image_in, device, l_norm=1, lambda_gp=1e-1):
#     """Calculate the image gradient penalty loss. New contribution from our volume generation work

#     Arguments:
#         image_out               -- target image with which to calculate gradients
#         image_in                -- input image
#     Returns gradient penalty and tensor of image gradients
#     """
#     if lambda_gp > 0.0:
#         gradients = torch.autograd.grad(outputs=image_out, inputs=image_in,
#                                         grad_outputs=torch.ones(image_out.size()).to(device),
#                                         create_graph=True, retain_graph=True, only_inputs=True)
#         gradients = gradients[0].view(image_in.size(0), -1)  # flat the data
#         gradient_penalty = (gradients + 1e-16).norm(l_norm, dim=1).mean() * lambda_gp        # added eps
#         return gradient_penalty, gradients
#     else:
#         return 0.0, None

def cal_image_grad_penalty(image_out, image_in, device):
    pass


class ResnetGenerator(nn.Module):


    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), #对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),  #归一化层
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out




# class SRResnetGenerator(nn.Module):
#     def __init__(
#             self,
#             input_nc,
#             output_nc,
#             mult,
#             ngf=64,
#             #norm_layer=nn.BatchNorm2d,  # 对输入batch的每一个特征通道进行normalize
#             norm_layer=nn.InstanceNorm2d,
#             use_dropout=False,
#             n_blocks=6,
#             padding_type='reflect'):
#         assert (n_blocks >= 0)
#         super(SRResnetGenerator, self).__init__()
#
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#
#         premodel = [nn.ReflectionPad2d(4),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
#                     nn.Conv2d(input_nc, ngf, kernel_size=9, padding=0, bias=use_bias),
#                     nn.PReLU()]  # nn.ReLU(True)
#
#         sft_block = []
#         sft_block += [nn.ReflectionPad2d(4),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
#                       nn.Conv2d(input_nc, ngf, kernel_size=9, padding=0, bias=use_bias),
#                       nn.PReLU()]  # nn.ReLU(True)
#         for i in range(2):
#             sft_block += [nn.ReflectionPad2d(4),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
#                           nn.Conv2d(ngf, ngf, kernel_size=9, padding=0, bias=use_bias),
#                           nn.PReLU()]  # nn.ReLU(True)
#         sft_block += [nn.ReflectionPad2d(4),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
#                       nn.Conv2d(ngf, ngf, kernel_size=9, padding=0, bias=use_bias)]
#
#         model1=[]
#         model1 += [
#             SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                           use_bias=use_bias)]
#         model11 = []
#         model11 += [
#             SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                            use_bias=use_bias)]
#
#         model2 = []
#         model2 += [
#             SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                           use_bias=use_bias)]
#         model22 = []
#         model22 += [
#             SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                            use_bias=use_bias)]
#
#         model3 = []
#         model3 += [
#             SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                           use_bias=use_bias)]
#         model33 = []
#         model33 += [
#             SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                            use_bias=use_bias)]
#
#         model4 = []
#         model4 += [
#             SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                           use_bias=use_bias)]
#         model44 = []
#         model44 += [
#             SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                            use_bias=use_bias)]
#
#         model5 = []
#         model5 += [
#             SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                           use_bias=use_bias)]
#         model55 = []
#         model55 += [
#             SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                            use_bias=use_bias)]
#
#         model6 = []
#         model6 += [
#             SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                           use_bias=use_bias)]
#         model66 = []
#         model66 += [
#             SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                            use_bias=use_bias)]
#
#         model7 = []
#         model7 += [
#             SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                           use_bias=use_bias)]
#         model77 = []
#         model77 += [
#             SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                            use_bias=use_bias)]
#
#         model8 = []
#         model8 += [
#             SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                           use_bias=use_bias)]
#         model88 = []
#         model88 += [
#             SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                            use_bias=use_bias)]
#
#
#         model9 = []
#
#         model9 += [
#             SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                               use_bias=use_bias)]
#         model99 = []
#         model99 += [
#             SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                            use_bias=use_bias)]
#
#         model =[]
#         # model10 += [
#         #     SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#         #                    use_bias=use_bias)]
#         model += [nn.ReflectionPad2d(1), nn.Conv2d(ngf, ngf, kernel_size=3, padding=0, bias=use_bias), norm_layer(ngf)]
#
#
#         # model = []
#         # for _ in range(n_blocks):  # add ResNet blocks
#         #     model += [SRResnetBlock(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#         #                             use_bias=use_bias)]
#         # model += [nn.ReflectionPad2d(1), nn.Conv2d(ngf, ngf, kernel_size=3, padding=0, bias=use_bias), norm_layer(ngf)]
#         # nn.ReflectionPad2d(1)上采样
#
#
#
#         upsample = []
#         for i in range(int(np.log2(mult))):  # add upsampling layers
#             upsample += [nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias),
#                          nn.PixelShuffle(2),
#                          nn.PReLU()]  # nn.ReLU(True)
#         upsample += [nn.ReflectionPad2d(3)]
#         upsample += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
#         upsample += [nn.Tanh()]
#
#         self.premodel = nn.Sequential(*premodel)
#         self.sft_block = nn.Sequential(*sft_block)
#         self.model1 = nn.Sequential(*model1)
#         self.model11 = nn.Sequential(*model11)
#         self.model2 = nn.Sequential(*model2)
#         self.model22 = nn.Sequential(*model22)
#         self.model3 = nn.Sequential(*model3)
#         self.model33 = nn.Sequential(*model33)
#         self.model4 = nn.Sequential(*model4)
#         self.model44 = nn.Sequential(*model44)
#         self.model5 = nn.Sequential(*model5)
#         self.model55 = nn.Sequential(*model55)
#         self.model6 = nn.Sequential(*model6)
#         self.model66 = nn.Sequential(*model66)
#         self.model7 = nn.Sequential(*model7)
#         self.model77 = nn.Sequential(*model77)
#         self.model8 = nn.Sequential(*model8)
#         self.model88 = nn.Sequential(*model88)
#         self.model9 = nn.Sequential(*model9)
#         self.model99 = nn.Sequential(*model99)
#         self.model = nn.Sequential(*model)
#         self.upsample = nn.Sequential(*upsample)
#
#     def forward(self, input):
#         """Standard forward"""
#         input1, input2 = torch.chunk(input, 2, 0)
#         a = self.premodel(input1)
#         b = self.sft_block(input2)
#         x = torch.unsqueeze(a,dim=0)
#         y = torch.unsqueeze(b, dim=0)
#         z =torch.cat([x, y], 0)
#         m1 = self.model1(z)
#         x = torch.unsqueeze(m1, dim=0)
#         z = torch.cat([x, y], 0)
#         m11 = a + self.model11(z)
#
#         x = torch.unsqueeze(m11,dim=0)
#         z = torch.cat([x, y], 0)
#         m2 = self.model2(z)
#         x = torch.unsqueeze(m2, dim=0)
#         z = torch.cat([x, y], 0)
#         m22 = m11 + self.model22(z)
#
#         x = torch.unsqueeze(m22, dim=0)
#         z = torch.cat([x, y], 0)
#         m3 = self.model3(z)
#         x = torch.unsqueeze(m3, dim=0)
#         z = torch.cat([x, y], 0)
#         m33 = m22 + self.model33(z)
#
#         x = torch.unsqueeze(m33, dim=0)
#         z = torch.cat([x, y], 0)
#         m4 = self.model4(z)
#         x = torch.unsqueeze(m4, dim=0)
#         z = torch.cat([x, y], 0)
#         m44 = m33 + self.model44(z)
#
#         x = torch.unsqueeze(m44, dim=0)
#         z = torch.cat([x, y], 0)
#         m5 = self.model5(z)
#         x = torch.unsqueeze(m5, dim=0)
#         z = torch.cat([x, y], 0)
#         m55 = m44 + self.model55(z)
#
#         x = torch.unsqueeze(m55, dim=0)
#         z = torch.cat([x, y], 0)
#         m6 = self.model6(z)
#         x = torch.unsqueeze(m6, dim=0)
#         z = torch.cat([x, y], 0)
#         m66 = m55 + self.model66(z)
#
#         x = torch.unsqueeze(m66, dim=0)
#         z = torch.cat([x, y], 0)
#         m7 = self.model7(z)
#         x = torch.unsqueeze(m7, dim=0)
#         z = torch.cat([x, y], 0)
#         m77 = m66 + self.model77(z)
#
#         x = torch.unsqueeze(m77, dim=0)
#         z = torch.cat([x, y], 0)
#         m8 = self.model8(z)
#         x = torch.unsqueeze(m8, dim=0)
#         z = torch.cat([x, y], 0)
#         m88 = m77 + self.model88(z)
#
#         x = torch.unsqueeze(m88, dim=0)
#         z = torch.cat([x, y], 0)
#         m9 = self.model9(z)
#         x = torch.unsqueeze(m9, dim=0)
#         z = torch.cat([x, y], 0)
#         m99 = m88 + self.model99(z)
#         m10 = b * m99 +b
#         m11 = a+m10
#         m=self.model(m11)
#         out = self.upsample(m)
#         return out
class SRResnetGenerator(nn.Module):
    def __init__(
            self,
            input_nc,
            output_nc,
            mult,
            ngf=64,
            # norm_layer=nn.BatchNorm2d,  # 对输入batch的每一个特征通道进行normalize
            norm_layer=nn.InstanceNorm2d,
            use_dropout=False,
            n_blocks=6,
            padding_type='reflect'):
        assert (n_blocks >= 0)
        super(SRResnetGenerator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        LRmodel1=[]

        LRmodel1 += [nn.ReflectionPad2d(1),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                    nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0, bias=use_bias),
                    nn.PReLU()]
        LRmodel1 += [nn.ReflectionPad2d(1),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=3, padding=0, bias=use_bias),
                     nn.PReLU()]
        ssmodel1=[]
        ssmodel1 += [nn.ReflectionPad2d(1),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                    nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0, bias=use_bias),
                    nn.PReLU()]
        ssmodel1 += [nn.ReflectionPad2d(1),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=3, padding=0, bias=use_bias)
                     ]
        depthmodel1 = []
        depthmodel1 += [nn.ReflectionPad2d(1),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0, bias=use_bias),
                     nn.PReLU()]
        depthmodel1 += [nn.ReflectionPad2d(1),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=3, padding=0, bias=use_bias)
                     ]
        model1 = []
        model1 += [
            SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=3)]
        model11 = []
        model11 += [
            SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=3)]
        model111 = []
        model111 += [
            SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=3)]
        model1111 = []
        model1111 += [
            SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=3)]
        
        LRmodel2=[]

        LRmodel2 += [nn.ReflectionPad2d(2),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                    nn.Conv2d(ngf, ngf, kernel_size=5, padding=0, bias=use_bias),
                    nn.PReLU()]
        LRmodel2 += [nn.ReflectionPad2d(2),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=5, padding=0, bias=use_bias),
                     nn.PReLU()]
        ssmodel2=[]
        ssmodel2 += [nn.ReflectionPad2d(2),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                    nn.Conv2d(ngf, ngf, kernel_size=5, padding=0, bias=use_bias),
                    nn.PReLU()]
        ssmodel2 += [nn.ReflectionPad2d(2),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=5, padding=0, bias=use_bias)
                     ]
        depthmodel2 = []
        depthmodel2 += [nn.ReflectionPad2d(2),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=5, padding=0, bias=use_bias),
                     nn.PReLU()]
        depthmodel2 += [nn.ReflectionPad2d(2),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=5, padding=0, bias=use_bias)
                     ]
        model2 = []
        model2 += [
            SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=5)]
        model22 = []
        model22 += [
            SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=5)]
        model222 = []
        model222 += [
            SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=5)]
        model2222 = []
        model2222 += [
            SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=5)]
        
        LRmodel3=[]

        LRmodel3 += [nn.ReflectionPad2d(3),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                    nn.Conv2d(ngf, ngf, kernel_size=7, padding=0, bias=use_bias),
                    nn.PReLU()]
        LRmodel3 += [nn.ReflectionPad2d(3),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=7, padding=0, bias=use_bias),
                     nn.PReLU()]
        ssmodel3=[]
        ssmodel3 += [nn.ReflectionPad2d(3),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                    nn.Conv2d(ngf, ngf, kernel_size=7, padding=0, bias=use_bias),
                    nn.PReLU()]
        ssmodel3 += [nn.ReflectionPad2d(3),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=7, padding=0, bias=use_bias)
                     ]
        depthmodel3 = []
        depthmodel3 += [nn.ReflectionPad2d(3),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=7, padding=0, bias=use_bias),
                     nn.PReLU()]
        depthmodel3 += [nn.ReflectionPad2d(3),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=7, padding=0, bias=use_bias)
                     ]
        model3 = []
        model3 += [
            SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=7)]
        model33 = []
        model33 += [
            SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=7)]
        model333 = []
        model333 += [
            SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=7)]
        model3333 = []
        model3333 += [
            SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=7)]
        
        LRmodel4=[]

        LRmodel4 += [nn.ReflectionPad2d(4),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                    nn.Conv2d(ngf, ngf, kernel_size=9, padding=0, bias=use_bias),
                    nn.PReLU()]
        LRmodel4 += [nn.ReflectionPad2d(4),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=9, padding=0, bias=use_bias),
                     nn.PReLU()]
        ssmodel4=[]
        ssmodel4 += [nn.ReflectionPad2d(4),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                    nn.Conv2d(ngf, ngf, kernel_size=9, padding=0, bias=use_bias),
                    nn.PReLU()]
        ssmodel4 += [nn.ReflectionPad2d(4),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=9, padding=0, bias=use_bias)
                     ]
        depthmodel4 = []
        depthmodel4 += [nn.ReflectionPad2d(4),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=9, padding=0, bias=use_bias),
                     nn.PReLU()]
        depthmodel4 += [nn.ReflectionPad2d(4),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=9, padding=0, bias=use_bias)
                     ]
        model4 = []
        model4 += [
            SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=9)]
        model44 = []
        model44 += [
            SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=9)]
        model444 = []
        model444 += [
            SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=9)]
        model4444 = []
        model4444 += [
            SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=9)]
        
        LRmodel5=[]

        LRmodel5 += [nn.ReflectionPad2d(5),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                    nn.Conv2d(ngf, ngf, kernel_size=11, padding=0, bias=use_bias),
                    nn.PReLU()]
        LRmodel5 += [nn.ReflectionPad2d(5),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=11, padding=0, bias=use_bias),
                     nn.PReLU()]
        ssmodel5=[]
        ssmodel5 += [nn.ReflectionPad2d(5),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                    nn.Conv2d(ngf, ngf, kernel_size=11, padding=0, bias=use_bias),
                    nn.PReLU()]
        ssmodel5 += [nn.ReflectionPad2d(5),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=11, padding=0, bias=use_bias)
                     ]
        depthmodel5 = []
        depthmodel5 += [nn.ReflectionPad2d(5),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=11, padding=0, bias=use_bias),
                     nn.PReLU()]
        depthmodel5 += [nn.ReflectionPad2d(5),  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充。
                     nn.Conv2d(ngf, ngf, kernel_size=11, padding=0, bias=use_bias)
                     ]
        model5 = []
        model5 += [
            SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=11)]
        model55 = []
        model55 += [
            SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=11)]
        model555 = []
        model555 += [
            SRResnetBlock1(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=11)]
        model5555 = []
        model5555 += [
            SRResnetBlock2(input_nc, ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                           use_bias=use_bias, size=11)]

        model = []

        model += [nn.ReflectionPad2d(1), nn.Conv2d(ngf, ngf, kernel_size=3, padding=0, bias=use_bias), norm_layer(ngf)]

        upsample = []
        for i in range(int(np.log2(mult))):  # add upsampling layers
            upsample += [nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1, bias=use_bias),
                         nn.PixelShuffle(2),
                         nn.PReLU()]  # nn.ReLU(True)
        upsample += [nn.ReflectionPad2d(3)]
        upsample += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        upsample += [nn.Tanh()]

        self.LRmodel1 = nn.Sequential(*LRmodel1)
        self.ssmodel1 = nn.Sequential(*ssmodel1)
        self.depthmodel1 = nn.Sequential(*depthmodel1)
        self.model1 = nn.Sequential(*model1)
        self.model11 = nn.Sequential(*model11)
        self.model111 = nn.Sequential(*model111)
        self.model1111 = nn.Sequential(*model1111)
        self.LRmodel2 = nn.Sequential(*LRmodel2)
        self.ssmodel2 = nn.Sequential(*ssmodel2)
        self.depthmodel2 = nn.Sequential(*depthmodel2)
        self.model2 = nn.Sequential(*model2)
        self.model22 = nn.Sequential(*model22)
        self.model222 = nn.Sequential(*model222)
        self.model2222 = nn.Sequential(*model2222)
        self.LRmodel3 = nn.Sequential(*LRmodel3)
        self.ssmodel3 = nn.Sequential(*ssmodel3)
        self.depthmodel3 = nn.Sequential(*depthmodel3)
        self.model3 = nn.Sequential(*model3)
        self.model33 = nn.Sequential(*model33)
        self.model333 = nn.Sequential(*model333)
        self.model3333 = nn.Sequential(*model3333)
        self.LRmodel4 = nn.Sequential(*LRmodel4)
        self.ssmodel4 = nn.Sequential(*ssmodel4)
        self.depthmodel4 = nn.Sequential(*depthmodel4)
        self.model4 = nn.Sequential(*model4)
        self.model44 = nn.Sequential(*model44)
        self.model444 = nn.Sequential(*model444)
        self.model4444 = nn.Sequential(*model4444)
        self.LRmodel5 = nn.Sequential(*LRmodel5)
        self.ssmodel5 = nn.Sequential(*ssmodel5)
        self.depthmodel5 = nn.Sequential(*depthmodel5)
        self.model5 = nn.Sequential(*model5)
        self.model55 = nn.Sequential(*model55)
        self.model555 = nn.Sequential(*model555)
        self.model5555 = nn.Sequential(*model5555)
        self.model = nn.Sequential(*model)
        self.upsample = nn.Sequential(*upsample)

    def forward(self, input):
        """Standard forward"""
        input1, input2, input3 = torch.chunk(input, 3, 0)
        a1 = self.LRmodel1(input1)
        b1 = self.ssmodel1(input2)
        c1 = self.depthmodel1(input3)
        x = torch.unsqueeze(a1, dim=0)
        y = torch.unsqueeze(b1, dim=0)
        w = torch.unsqueeze(c1, dim=0)
        z1 = torch.cat([x, y], 0)
        z2 = torch.cat([x, w], 0)
        m1 = self.model1(z1)
        m11 = self.model111(z2)
        x = torch.unsqueeze(m1, dim=0)
        w = torch.unsqueeze(m11, dim=0)
        z1 = torch.cat([x, y], 0)
        z2 = torch.cat([x, w], 0)
        m111 = a1 + (self.model11(z1)+self.model1111(z2))*0.5

        a2 = self.LRmodel2(m111)
        b2 = self.ssmodel2(b1)
        c2 = self.depthmodel2(c1)
        x = torch.unsqueeze(a2, dim=0)
        y = torch.unsqueeze(b2, dim=0)
        w = torch.unsqueeze(c2, dim=0)
        z1 = torch.cat([x, y], 0)
        z2 = torch.cat([x, w], 0)
        m2 = self.model2(z1)
        m22 = self.model222(z2)
        x = torch.unsqueeze(m2, dim=0)
        w = torch.unsqueeze(m22, dim=0)
        z1 = torch.cat([x, y], 0)
        z2 = torch.cat([x, w], 0)
        m222 = a2 + (self.model22(z1) + self.model2222(z2)) * 0.5

        a3 = self.LRmodel3(m222)
        b3 = self.ssmodel3(b2)
        c3 = self.depthmodel3(c2)
        x = torch.unsqueeze(a3, dim=0)
        y = torch.unsqueeze(b3, dim=0)
        w = torch.unsqueeze(c3, dim=0)
        z1 = torch.cat([x, y], 0)
        z2 = torch.cat([x, w], 0)
        m3 = self.model3(z1)
        m33 = self.model333(z2)
        x = torch.unsqueeze(m3, dim=0)
        w = torch.unsqueeze(m33, dim=0)
        z1 = torch.cat([x, y], 0)
        z2 = torch.cat([x, w], 0)
        m333 = a2 + (self.model33(z1) + self.model3333(z2)) * 0.5

        a4 = self.LRmodel4(m333)
        b4 = self.ssmodel4(b3)
        c4 = self.depthmodel4(c3)
        x = torch.unsqueeze(a4, dim=0)
        y = torch.unsqueeze(b4, dim=0)
        w = torch.unsqueeze(c4, dim=0)
        z1 = torch.cat([x, y], 0)
        z2 = torch.cat([x, w], 0)
        m4 = self.model4(z1)
        m44 = self.model444(z2)
        x = torch.unsqueeze(m4, dim=0)
        w = torch.unsqueeze(m44, dim=0)
        z1 = torch.cat([x, y], 0)
        z2 = torch.cat([x, w], 0)
        m444 = a2 + (self.model44(z1) + self.model4444(z2)) * 0.5

        a5 = self.LRmodel5(m444)
        b5 = self.ssmodel5(b4)
        c5 = self.depthmodel5(c4)
        x = torch.unsqueeze(a5, dim=0)
        y = torch.unsqueeze(b5, dim=0)
        w = torch.unsqueeze(c5, dim=0)
        z1 = torch.cat([x, y], 0)
        z2 = torch.cat([x, w], 0)
        m5 = self.model5(z1)
        m55 = self.model555(z2)
        x = torch.unsqueeze(m5, dim=0)
        w = torch.unsqueeze(m55, dim=0)
        z1 = torch.cat([x, y], 0)
        z2 = torch.cat([x, w], 0)
        m555 = a2 + (self.model55(z1) + self.model5555(z2)) * 0.5

        m = self.model(m555)
        out = self.upsample(m)
        return out


class SRResnetBlock1(nn.Module):
    """Define a Resnet block"""

    def __init__(self,input_nc, dim, padding_type, norm_layer, use_dropout, use_bias, size):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(SRResnetBlock1, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,size)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias,size):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(int(size/2))]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(int(size/2))]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=size, padding=p, bias=use_bias), norm_layer(dim),
                       nn.PReLU()]  # nn.ReLU(True)
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]


        # p = 0
        # if padding_type == 'reflect':
        #     conv_block += [nn.ReflectionPad2d(1)]
        # elif padding_type == 'replicate':
        #     conv_block += [nn.ReplicationPad2d(1)]
        # elif padding_type == 'zero':
        #     p = 1
        # else:
        #     raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        # conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)



    def forward(self, x):
        """Forward function (with skip connections)"""
        input1, input2 = torch.chunk(x, 2, 0)
        input1 = torch.squeeze(input1, dim=0)
        input2 = torch.squeeze(input2, dim=0)

        out1 = input2*input1+input2
        out = self.conv_block(out1)
        return out


class SRResnetBlock2(nn.Module):
    """Define a Resnet block"""

    def __init__(self, input_nc, dim, padding_type, norm_layer, use_dropout, use_bias, size):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(SRResnetBlock2, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, size)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, size):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        # p = 0
        # if padding_type == 'reflect':
        #     conv_block += [nn.ReflectionPad2d(1)]
        # elif padding_type == 'replicate':
        #     conv_block += [nn.ReplicationPad2d(1)]
        # elif padding_type == 'zero':
        #     p = 1
        # else:
        #     raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        #
        # conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim),
        #                nn.PReLU()]  # nn.ReLU(True)
        # if use_dropout:
        #     conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(int(size/2))]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(int(size/2))]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=size, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        input1, input2 = torch.chunk(x, 2, 0)
        input1 = torch.squeeze(input1, dim=0)
        input2 = torch.squeeze(input2, dim=0)

        out1 = input2 * input1 + input2
        out = self.conv_block(out1)
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
#             print("++++++++x.shape+++")
#             print(x.shape)
            return torch.cat([x, self.model(x)], 1)


class LinearModel(nn.Module):
    '''
    One-layer linear neural network for the least-squares regression model. 
        Sums all pixels in TXM image to predict center pixel of SEM image
    '''

    def __init__(self, nonlinear=False):
        super().__init__()

        self.nonlinear = nonlinear # Store if model is nonlinear

        if nonlinear:
            self.gamma = nn.Parameter(torch.ones(1)) # gamma filter parameter

        self.filter = nn.Conv2d(1, 1, 15, padding=7) # Forward convolution function

    def forward(self, x):
        if self.nonlinear:
            x = torch.pow(x+1e-16, self.gamma)
        out = self.filter(x)
        return out


class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
