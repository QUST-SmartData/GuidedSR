from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')#在屏幕上显示训练结果的频率
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
                   #如果为正值，则在单个visdom web面板中显示所有图像，每行显示一定数量的图像。
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')#web显示的visdom服务器
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')#visdom显示环境名称（默认为“main”）
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')#web显示器的visdom端口
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')#训练结果保存到html频率
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')#训练结果在控制台显示频率
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
               #不要将中间培训结果保存到这个地址
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')#保存最新结果的频率
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')#在时间段结束时保存检查点的频率
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')#是否通过迭代保存模型
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')#继续培训：加载最新模型
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
                   #起始epoch计数
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')#阶段 模式
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')#iter的起始学习率
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')#iter将学习率线性衰减至零
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')#优化器的初始学习率
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
                 #存储先前生成的图像的图像缓冲区的大小
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')#学习率模式
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')#每lr_decay_iters迭代乘以伽马
        parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 regularization for the generator network')#生成网络l2正则化

        self.isTrain = True
        return parser
