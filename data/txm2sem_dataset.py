"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import glob
import re
import os
import random
from util import util
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


class Txm2semDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--aligned', type=eval, default=True, help='optionally use aligned or unaligned image patches') #（可选）使用对齐或未对齐的图像面片
        parser.add_argument('--eval_mode', type=eval, default=False, help='determines whether dataset has fixed or random indices')#确定数据集具有固定索引还是随机索引
        parser.add_argument('--patch_size', type=int, default=128, help='image patch size when performing subsampling')
        parser.add_argument('--txm_dir', type=str, default='txm', help='directory containing TXM images')
        parser.add_argument('--sem_dir', type=str, default='sem', help='directory containing SEM images')
        parser.add_argument('--ss_dir', type=str, default='ss', help='directory containing SS images')
        parser.add_argument('--charge_dir', type=str, default='charge', help='directory containing TXM images')
        parser.add_argument('--lowdens_dir', type=str, default='lowdensity', help='directory containing TXM images')
        parser.add_argument('--highdens_dir', type=str, default='highdensity', help='directory containing TXM images')
        parser.add_argument('--num_train', type=int, default=10000, help='number of image patches to sample for training set')
        parser.add_argument('--x_misalign', type=int, default=0, help='misalignment factor in x-direction to offset input TXM images')#x方向偏移输入TXM图像的未对准系数
        parser.add_argument('--y_misalign', type=int, default=0, help='misalignment factor in y-direction to offset input TXM images')
        #新添加
        parser.add_argument('--save_name',type=str,default='save_name',help='option to save the TXM image stack during evaluation (automatically true if preprocessing TXM images)')

        #parser.set_defaults(max_dataset_size=10000, new_dataset_option=2.0, num_test=100)  # specify dataset-specific default values指定数据集特定的默认值
        
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        self.patch_size = opt.patch_size
        self.aligned = opt.aligned
        self.eval_mode = opt.eval_mode
        self.full_slice = opt.full_slice
        self.x_misalign = opt.x_misalign
        self.y_misalign = opt.y_misalign

        if opt.save_name != '':
            opt.save_name = '_' + opt.save_name

        # set whether to perform downsampling and to include   设置是否执行下采样并包括：
        if opt.model in ['srcnn', 'srgan']:
            self.downsample_factor = opt.downsample_factor
            if hasattr(opt, 'd_condition'):
                self.include_original_res = opt.d_condition
            else:
                self.include_original_res = False
        else:
            self.downsample_factor = None
        
        # get images for dataset;
        img_nums = []
        TXM = []
        SEM = []
        SS=[]
        charges = []
        lowdens = []
        highdens = []
        
        if opt.isTrain:
            if opt.eval_mode:
                base_img_dir = './images/validation/'
            else:
                base_img_dir = './images/train/'
            base_save_imgs_dir = os.path.join(opt.checkpoints_dir, opt.name, 'sample_imgs')
        else:
            if opt.phase =='val':
                base_img_dir = './images/validation/'
                base_save_imgs_dir = os.path.join(opt.checkpoints_dir, opt.name, 'sample_imgs')
            else:
                base_img_dir = './images/test/'
                base_save_imgs_dir = os.path.join(opt.results_dir, opt.name)

        txm_dir = os.path.join(base_img_dir + opt.txm_dir + opt.sample_name, '')
        sem_dir = os.path.join(base_img_dir + opt.sem_dir + opt.sample_name, '')
        ss_dir = os.path.join(base_img_dir + opt.ss_dir + opt.sample_name, '')
        charge_dir = os.path.join(base_img_dir + opt.charge_dir + opt.sample_name, '')
        lowdens_dir = os.path.join(base_img_dir + opt.lowdens_dir + opt.sample_name, '')
        highdens_dir = os.path.join(base_img_dir + opt.highdens_dir + opt.sample_name, '')

        for f in glob.glob(txm_dir+'*.tif'):  #glob.glob()查找符合特定规则的文件路径名
            img_nums.append(max(map(int, re.findall('\d+', f))))
            imnum = str(img_nums[-1]).zfill(3)
            TXM.append(np.asarray(Image.open(txm_dir+'TXM'+imnum+'.tif').convert('L')))
            SEM.append(np.asarray(Image.open(sem_dir+'SEM'+imnum+'.tif').convert('L')))
            charges.append(np.asarray(Image.open(charge_dir+'charge'+imnum+'.tif')))
            lowdens.append(np.asarray(Image.open(lowdens_dir+'SEM_dark'+imnum+'.tif')))
            highdens.append(np.asarray(Image.open(highdens_dir+'SEM_light'+imnum+'.tif')))
            SS.append(np.asarray(Image.open(ss_dir+'SS'+imnum+'.tif').convert('L')))
        
        # Sort according to slice number  根据切片编号排序
        sort_inds = np.argsort(img_nums)
        self.txm = [TXM[i] for i in sort_inds]
        self.sem = [SEM[i] for i in sort_inds]
        self.ss = [SS[i] for i in sort_inds]
        self.charges = [charges[i] for i in sort_inds]
        self.lowdens = [lowdens[i] for i in sort_inds]
        self.highdens = [highdens[i] for i in sort_inds]

        # Define the default transform function from base transform funtion. 
        if opt.eval_mode:  #false  不进入
            opt.no_flip = True
        self.txm_transform = get_transform(opt)
        self.sem_transform = get_transform(opt, sem_xform=True)
        self.ss_transform = get_transform(opt)

        if self.full_slice:  #false  不会进入
            self.length = len(self.txm)
            self.sem_fake_save_dir = os.path.join(base_save_imgs_dir, 'sem_fake_fullslice/')
            util.mkdirs([self.sem_fake_save_dir])
        
        else:
            if opt.isTrain:  #true
                self.length = opt.num_train  #10000
            else:
                self.length = opt.num_test

            # Get patch indices and save subset of patches
            self.txm_save_dir = os.path.join(base_save_imgs_dir, opt.txm_dir + opt.save_name, '')
            self.sem_save_dir = os.path.join(base_save_imgs_dir, opt.sem_dir + opt.save_name, '')
            self.ss_save_dir = os.path.join(base_save_imgs_dir, opt.ss_dir + opt.save_name, '')
            self.sem_fake_save_dir = os.path.join(base_save_imgs_dir, 'sem_fake' + opt.save_name, '')
            self.charge_save_dir = os.path.join(base_save_imgs_dir, opt.charge_dir)
            self.lowdens_save_dir = os.path.join(base_save_imgs_dir, opt.lowdens_dir)
            self.highdens_save_dir = os.path.join(base_save_imgs_dir, opt.highdens_dir)
            util.mkdirs([self.txm_save_dir, self.sem_save_dir, self.ss_save_dir, self.sem_fake_save_dir, self.charge_save_dir, self.lowdens_save_dir, self.highdens_save_dir])
            
            # Sample fixed patch indices if set to evaluation mode
            if self.eval_mode:  #false  不会进入
                np.random.seed(999)
                random.seed(999)
                self.indices = []
                for i in range(self.length):
                    inds_temp = self.get_aligned_patch_inds()
                    self.indices.append(inds_temp)

                    txm_patch, sem_patch, ss_patch = self.get_patch(i)
                    txm_patch, sem_patch, ss_patch = util.tensor2im(torch.unsqueeze(txm_patch,0)), util.tensor2im(torch.unsqueeze(sem_patch,0)), util.tensor2im(torch.unsqueeze(ss_patch,0))
                    # sem_patch = (sem_patch.astype(np.float) * 2.0 / 255.0 - 1)*255.0
                    # sem_patch = sem_patch.astype(np.uint8)
                    # txm_patch = (txm_patch.astype(np.float) * 2.0 / 255.0 - 1)*255.0
                    # txm_patch = txm_patch.astype(np.uint8)

                    txm_path = self.txm_save_dir + str(i).zfill(3) + '.png'
                    sem_path = self.sem_save_dir + str(i).zfill(3) + '.png'
                    ss_path = self.ss_save_dir + str(i).zfill(3) + '.png'
                    
                    util.save_image(txm_patch, txm_path)
                    util.save_image(sem_patch, sem_path)

                    # Save charge mask and low/high density segmentations
                    xcoord, ycoord, zcoord = inds_temp
                    charge_patch = 255*self.charges[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]
                    lowdens_patch = 255*self.lowdens[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]
                    highdens_patch = 255*self.highdens[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]

                    charge_path = self.charge_save_dir + str(i).zfill(3) + '.png'
                    lowdens_path = self.lowdens_save_dir + str(i).zfill(3) + '.png'
                    highdens_path = self.highdens_save_dir + str(i).zfill(3) + '.png'

                    util.save_image(charge_patch, charge_path)
                    util.save_image(lowdens_patch, lowdens_path)
                    util.save_image(highdens_patch, highdens_path)
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """

        if self.full_slice:##不进入
            def xform_temp(x): return self.transform(Image.fromarray(x[...,:1024,:1024]))
            data_A, data_B, data_SS = xform_temp(self.txm[index]), xform_temp(self.sem[index]),  xform_temp(self.ss[index])
            A_paths, B_paths, SS_paths= self.sem_fake_save_dir + str(index).zfill(3) + '.png', self.sem_fake_save_dir + str(index).zfill(3) + '.png' + '.png', self.sem_fake_save_dir + str(index).zfill(3) + '.png'+ '.png' + '.png'
            A_orig = None

        else:
            data_A, data_B, data_SS, A_orig = self.get_patch(index, return_original=True)
            A_paths = self.txm_save_dir + str(index).zfill(3) + '.png' 
            B_paths = self.sem_fake_save_dir + str(index).zfill(3) + '.png'
            SS_paths = self.ss_save_dir + str(index).zfill(3) + '.png'

        # Data transformation needs to convert to tensor
        return {'A': data_A, 'B': data_B, 'SS': data_SS, 'A_orig': A_orig, 'A_paths': A_paths, 'B_paths': B_paths, 'SS_paths':SS_paths}


    def __len__(self):
        """Return the total number of images."""
        return self.length


    def get_patch(self, index, return_original=False):  ##切片操作
        '''
        Randomly sample patch from image stack
        ''' 
        if self.eval_mode:
            xcoord, ycoord, zcoord = self.indices[index] # Unpack indices
            # sem_patch = transforms.ToTensor()(Image.fromarray(self.sem[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]))
            # txm_patch = transforms.ToTensor()(Image.fromarray(self.txm[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]))
            #ss_trabsform =

            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            random.seed(seed) # apply this seed to img transforms
            sem_patch = self.sem_transform(Image.fromarray(self.sem[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]))
            random.seed(seed)
            txm_patch = self.txm_transform(Image.fromarray(self.txm[zcoord][xcoord+self.x_misalign:xcoord+self.patch_size+self.x_misalign, ycoord+self.y_misalign:ycoord+self.patch_size+self.y_misalign]))
            random.seed(seed)
            ss_patch = self.ss_transform(Image.fromarray(self.ss[zcoord][xcoord+self.x_misalign:xcoord+self.patch_size+self.x_misalign, ycoord+self.y_misalign:ycoord+self.patch_size+self.y_misalign]))

        else:
            indstemp = self.get_aligned_patch_inds()
            xcoord, ycoord, zcoord = indstemp
        
            # fix for performing same transform taken from: https://github.com/pytorch/vision/issues/9 
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            random.seed(seed) # apply this seed to img transforms
            sem_patch = self.sem_transform(Image.fromarray(self.sem[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]))
            random.seed(seed)
            txm_patch = self.txm_transform(Image.fromarray(self.txm[zcoord][xcoord+self.x_misalign:xcoord+self.patch_size+self.x_misalign, ycoord+self.y_misalign:ycoord+self.patch_size+self.y_misalign]))
            random.seed(seed)
            ss_patch = self.ss_transform(Image.fromarray(self.ss[zcoord][xcoord+self.x_misalign:xcoord+self.patch_size+self.x_misalign, ycoord+self.y_misalign:ycoord+self.patch_size+self.y_misalign]))

        if self.downsample_factor is not None:
            txm_processed = torch.unsqueeze(txm_patch, 1)
            txm_processed = F.interpolate(txm_processed, size=(int(self.patch_size/self.downsample_factor), int(self.patch_size/self.downsample_factor)))
            txm_processed = torch.unsqueeze(torch.squeeze(txm_processed), 0)
            ss_processed = torch.unsqueeze(ss_patch, 1)
            ss_processed = F.interpolate(ss_processed, size=(int(self.patch_size / self.downsample_factor), int(self.patch_size / self.downsample_factor)))
            ss_processed = torch.unsqueeze(torch.squeeze(ss_processed), 0)
        else:
            txm_processed = txm_patch
            ss_processed = ss_patch

        if return_original:
            return txm_processed, sem_patch, ss_processed, txm_patch
        else:
            return txm_patch, sem_patch, ss_patch


    def get_aligned_patch_inds(self):
        W, H = self.txm[0].shape
        good_patch = False
        
        while not good_patch:
            # Sample random coordinate
            xcoord = np.random.randint(0, high = W-self.patch_size)
            ycoord = np.random.randint(0, high = H-self.patch_size)
            zcoord = np.random.randint(0, high = len(self.txm))
            
            # Extract image patches from random coordinate
            sem_patch = self.sem[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]
            txm_patch = self.txm[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]
            ss_patch = self.ss[zcoord][xcoord:xcoord + self.patch_size, ycoord:ycoord + self.patch_size]
            charge_patch = 1-self.charges[zcoord][xcoord:xcoord+self.patch_size, ycoord:ycoord+self.patch_size]
            
            # Calculate mask of all black pixels across all three images
            mask = np.greater(charge_patch*sem_patch*txm_patch*ss_patch, 0)

            # Calculate number of pixels that are zero and are uncharged
            mask_prop =  np.sum(mask) / sem_patch.size
            uncharge_prop =  np.sum(charge_patch) / sem_patch.size
            
            # Check if patch is acceptable, if not resample
            if uncharge_prop > 0.95 and mask_prop > 0.75:
                good_patch = True

        return (xcoord, ycoord, zcoord)


    