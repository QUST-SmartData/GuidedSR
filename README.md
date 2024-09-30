# SemSR

# FractureSeg3D

This is the official repository for "A Super-resolution Framework with Semantic Guidance for Restoring Pore-Solid Interface Roughness to Enhance the Accuracy of Digital Rock Transport Properties". Please cite this work if you find this repository useful for your project.



The roughness of pore walls is a crucial factor in studying fluid flow within the pore space. Combining data from different imaging modalities and using deep learning-based super-resolution (SR) methods, a comprehensive view with intricate specific features would be obtained.The relationship between pore wall and pore space is typically representative of geological characterization, which distinguishes among different components. However, current SR methods often overlook geological component regions and incorporate various mechanisms that increase the model's weight and computational demands. To tackle these issues, we employ a Generative Adversarial Network and propose a semantic sharing mechanism to collaborate with the injection of geological characterization. In addition, matching low-resolution (LR) and high-resolution (HR) images is a major challenge. It is common practice to down-sample HR images to obtain pairs of LR images. However, the LR images obtained by these methods still contain lots of details, which weakens the model's generalization ability in real-world scenarios. Therefore, we developed a novel method that introduces intentional blurring noises and multi-sampling operations utilized during data augmentation. Finally, we compare our method with other state-of-the-art methods using proposed indicators to recover the true characteristics of the hole wall, proving the superiority of our method.



## Usage

The model is run through a command line interface. We strongly recommend installing [Anaconda](https://www.anaconda.com/products/individual) as it includes most of the packages needed for this code base, and the ``conda`` package management system is able to install almost everything required. 

### Setup

To begin, first install the dependencies listed here. The code requires the following packages listed below. Unless otherwise stated, these packages can be installed using ``conda`` or ``pip``
- ``torch``: install from the [PyTorch website](https://pytorch.org/)
- ``skimage``
- ``PIL``
- ``re``
- ``glob``
- ``dominate``
- ``visdom``: install with ``pip``

### Data 

Image data is expected to be stored using the following file structure for data loaders:
```
./images/
|
+--train/
|  +--txm/
|     +--[image number with three digits e.g. 000 or 015].tif
|  +--sem/
|     +--[image number].tif
|  +--charge/
|     +--[image number].tif
|  +--lowdensity/
|     +--[image number].tif
|  +--highdensity/
|     +--[image number].tif
|
+--val/
|  + ...
+--test/
|  + ...
|  +--txm_full_stack/
|     +--[image number in z-axis order, unrelated to the numbers of the aligned slices].tif
```

The data loaders rely on folders and filenames appearing in this specific form. If the images are not places in the correct folder, the dataloader will not be able to find them. Aligned image slices should appear with the same file names in the ``txm``, ``sem``, ``charge``, ``lowdensity``, and ``highdensity`` folders. This is how the code is able to track which slices are aligned with which. The ``txm_full_stack`` folder in the ``test/`` contains TXM images from a contiguous volume where each slice is numbered according to its slice number in the z-axis. 

Dataset files can be found in the ``./data/`` folder. The framework implements four data loaders depending on the application. The specific dataset to use is selected with the ``--dataset`` option during training and testing. 

**TXM2SEM**

The ``txm2sem_dataset.py`` file contains the main dataset for this framework. 

Command line options specific to this data loader are:
- ``--aligned``: optionally use aligned or unaligned image patches
- ``--eval_mode``: determines whether dataset has fixed indices (for evaluation) or random (for sampling during training)
- ``--patch_size``: image patch size when performing subsampling
- ``--txm_dir``: TXM image directory. This and all image directories below can be controlled using command line options but this is highly discouraged.
- ``--sem_dir``: SEM image directory
- ``--charge_dir``: charge region segmentation directory
- ``--lowdens_dir``: low density region segmentation directory
- ``--highdens_dir``: high density region segmentation directory
- ``--num_train``: number of datapoints in each training epoch

**Image Repair**

The ``image_repair_data.py`` loader functions very similarly to the ``txm2sem_dataset.py`` loader, with the only major difference being the form of the data output during sampling. Command line options specific for this data loader are the same as for the ``txm2sem_dataset.py`` loader above.

**TXM2SEM3D**

The ``txm2sem3d_dataset.py`` file contains a short dataloader to load TXM image volumes from the test set folder. TXM images in the ``/test/txm_full_stack/`` folder should be full image slices (uncropped). The code uses the ``x_ind`` and ``y_ind`` arguments as the top-left corner of the image patch from each slice. In this way, the subvolume to process can be controlled from the command line. 

Command line options specific to this data loader are:
- ``--patch_size``: image patch size when performing subsampling
   - This is also the size of the image volume in the z-direction. 
- ``--save_name``: directory to store the saved volume in the results folder
- ``--x_ind``: x-index for sampling image patches
- ``--y_ind``: y-index for sampling image patches 
- ``--z_ind``: z-index for sampling image patches 
   - The TXM image filenames are sorted descending in alpha-numeric order and the TXM volume is taken from successive image slices. ``z_ind`` will be the offset from the first file to start sampling. For example, if image filenames start with ``txm_full157.png`` and end with ``txm_full426.png``, then ``--z_ind 3`` will start evaluation with ``txm_full160.png``. If there are not enough files to start at the given ``z_ind`` and produce a volume of the size specified by ``patch_size``, then the code will return an error message and terminate without evaluating any images. 


### Training a model

The model is trained using the ``train.py`` script. For example, you can train a model using the following code: 

``python train.py --name srcnn_imgreg --model srcnn --netG sr_resnet_9blocks --niter 50 --niter_decay 25 --downsample_factor 2 --patch_size 128 --lambda_img_grad 1e-4``

This will train a SRCNN model with a 9 block SR-ResNet, decay the learning rate after 25 epochs, assume a 2x downsampling factor in the TXM images, train with 128x128 image patches, and uses a 10<sup>-4</sup> Jacobian penalty parameter. Each model has its own set of commands: 

**Feedforward CNN:** ``feedforward_model.py``
* ``--lambda_regression``: weight for the regression loss
* ``--regression_loss``: loss type for the regression (L2 or L1)
* ``--lambda_img_grad``: weight to image gradient penalty

**SR-CNN:** ``srcnn_model.py``
* ``--lambda_regression``: weight for the regression loss
* ``--regression_loss``: loss type for the regression (L2 or L1)
* ``--lambda_img_grad``: weight to image gradient penalty

**pix2pix CGAN:** ``pix2pix_model.py``
* ``--lambda_L1``: weight for L1 loss
* ``--lambda_img_grad``: weight to image gradient penalty
* ``--lambda_gp``: weight for gradient penalty in wgangp loss. Default to 0, must be entered manually as ``--gan_mode wgangp --lambda_gp 1e1`` (recommended value is 10).

**SRGAN:** ``srgan_model.py``
* ``--lambda_L1``: weight for L1 loss
* ``--lambda_img_grad``: weight to image gradient penalty
* ``--lambda_gp``: weight for gradient penalty in wgangp loss
* ``--d_condition``: if flag is passed, input TXM image will be passed to discriminator network along with generated SEM image. (This differs from the original SR-GAN model which only passes the generated image into the discriminator.) 

*Warning:* running ``train.py`` will clear all files in the ``./results/[model name]/`` directory and erase any previous results. Please be cautious not to delete any results by accidentally running the training script instead of the test script.


### Loading and testing a model 

The framework is automatically able to load a model based on the model name. To load a model, 

```
./checkpoints
|
+--/[model name]
   |
   +--[epoch #]_net_[net name].pth
   +--latest_net_[net name].pth

```

To test and evaluate a TXM-to-SEM image translation model, use:

``python test.py --name srgan_example --model srgan --netG sr_resnet_9blocks --downsample_factor 4 --patch_size 128``

During testing, The code will look in the checkpoints folder for a ``srgan_example`` folder, load the ``latest_net_G.pth`` in to a 4x SR-ResNet 9 block network, and evaluate the image patches and similarity metrics for the dataset. 

To evaluate for 3D volume translation, you must use the ``txm2sem3d`` dataset mode. Besides this, the command line argument is very similar:

``python test.py --name srgan_example --model srgan --netG sr_resnet_9blocks --downsample_factor 4 --patch_size 128 --dataset_mode txm2sem3d --x_ind 280 --y_ind 165 --save_name sem_predicted_volume ``

The framework will output the results in the following file structure:

```
./results 
|
+--/[model name]
   |
   +--/charge (charge region mask)
      +--[image patch number with three digits].png
      +--...
   +--/highdensity (high density region mask)
      +--...
   +--/lowdensity (low region mask)
      +--...
   +--/sem (ground truth SEM image)
      +--...
   +--/sem_fake (predicted SEM image)
      +--...
   +--/txm (input TXM image)
      +--...
   +--/volume_pred_test (predicted FIB-SEM volume slices)
      +--...
   +--/volume_txm_test (input TXM volume image slices)
      +--...
   +--eval_metrics.txt (summary of evaluation metrics)
```

The code saves the test set images every time it is run, but it is configured to produce the same test set subsampled image patches given the same test set slices. *Note:* the 3D dataset will also output quantitative image similarity metrics, but these results should be ignored during 3D volume generation. These results only appear because of difficulty changing the implementation.

We have two models from our research available for download: 
* SR GAN 4x with no image gradient penalty \[[Model](https://stanford.box.com/s/urj3uqwymkmx499zipu3w4ef99beypk1)\]
  * ``` python train.py --name srgan_no_reg --model srgan --netG sr_resnet_9blocks --niter 50 --niter_decay 25 --downsample_factor 4 --patch_size 128 --lambda_L1 100 ```
  * ``` python test.py --name srgan_no_reg --model srgan --netG sr_resnet_9blocks --downsample_factor 4 --patch_size 128 ```
  * ```  python test.py --name srgan_no_reg --model srgan --netG sr_resnet_9blocks --downsample_factor 4 --patch_size 128 --dataset_mode txm2sem3d --x_ind [x index] --y_ind [y index] --save_name [save name] ```

These models are ready to use or to test that your framework and data loader are working properly. 


### Command Line Interface Summary

Here we summarize the command line arguments used across all models. 

**General commands**

Basic parameters:
* ``--dataroot``: path to images. Should have subfolders ``training``, ``val``, and ``test``, which should each have subfolders ``TXM and ``SEM``
* ``--name``: ame of the experiment. It decides where to store samples and models.
* ``--gpu_ids``: gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU.
* ``--checkpoints_dir``: models are saved here (default: ``./checkpoints``).
* ``--downsample_factor``: factor by which to downsample synthetic data. Used with SISR models.

Model parameters:
* ``--model``: chooses which model to use. \[feedforward | srcnn | pix2pix | srgan \]
* ``--input_nc``: \# of input image channels (default=1, should not need to be changed)
* ``--output_nc``: \# of output image channels (default=1, should not need to be changed)
* ``--ngf``: \# of filters in the last conv layer of the generator, determines generator architecture
* ``--ndf``: \# of filters in the first conv layer of the discriminator, determines discriminator architecture
* ``--netD``: specify discriminator architecture \[ basic | n\_layers | pixel \]. The basic model is a 70x70 PatchGAN. n\_layers allows you to specify the layers in the discriminator. 
* ``--netG``: specify generator architecture \[ resnet\_9blocks | resnet\_6blocks | unet\_256 | unet\_128 | linearfilt \]
* ``--n_layers_D``: only used if netD==n\_layers
* ``--norm``: instance normalization or batch normalization \[ instance | batch | none \]
* ``--init_type``: network initialization \[ normal | xavier | kaiming | orthogonal \]
* ``--init_gain``: scaling factor for normal, xavier and orthogonal
* ``--no_dropout``: no dropout for the generator

Dataset parameters:
* ``--dataset_mode``: chooses how dataset loader \[ txm2sem | txm2sem3d | segmentation | image\_repair \]
* ``--direction``: AtoB or BtoA (where A is TXM and B is SEM, do not change)
* ``--serial_batches``: if flag is passed, takes images in order to make batches, otherwise takes them randomly
* ``--num_threads``: \# threads for loading data
* ``--batch_size``: input batch size
* ``--load_size``: scale images to this size (in data transform)
* ``--crop_size``: then crop images to this size (in data transform)
* ``--full_slice``: evaluate full image slices
* ``--max_dataset_size``: Maximum number of samples allowed per dataset. If the dataset directory contains more than max\_dataset\_size, only a subset is loaded.
* ``--preprocess``: scaling and cropping of images at load time \[ resize_and_crop | crop | scale\_width | scale\_width\_and\_crop | none \]
* ``--no_flip``: if specified, do not flip the images for data augmentation
* ``--display_winsize``: display window size for both visdom and HTML

Other parameters:
* ``--epoch``: which epoch to load (either epoch \# or set to ``latest`` to use latest cached model)
* ``--load_iter``: which iteration to load (if load\_iter > 0, the code will load models by ``--load_iter``; otherwise, the code will load models by ``--epoch``)
* ``--verbose``: if specified, print more debugging information
* ``--suffix``: customized suffix (name = name + suffix e.g. \{model\}\_\{netG\}\_size\{load_size\})

**Training-specific commands**

Display parameters (should not need to be altered):
* ``--display_freq``: frequency of showing training results on screen
* ``--display_ncols``: if positive, display all images in a single visdom web panel with certain number of images per row.
* ``--display_id``: window id of the web display
* ``--display_server``: visdom server of the web display
* ``--display_env``: visdom display environment name (default is "main")
* ``--display_port``: visdom port of the web display
* ``--update_html_freq``: frequency of saving training results to html
* ``--print_freq``: frequency of showing training results on console
* ``--no_html``: do not save intermediate training results to web checkpoint directory

Network saving and loading parameters:
* ``--save_latest_freq``: frequency of saving the latest results
* ``--save_epoch_freq``: frequency of saving checkpoints at the end of epochs
* ``--save_by_iter``: if flag passed, saves model by iteration
* ``--continue_train``: if flag passed, continue training by loading the latest model
* ``--epoch_count``: the starting epoch count, we save the model by <epoch\_count>, <epoch\_count>+<save\_latest\_freq>, ...
* ``--phase``: train, val, test, etc. Do not change this option to ensure proper behavior. 

Training parameters:
* ``--niter``: \# of iter at starting learning rate
* ``--niter_decay``: \# of iter to linearly decay learning rate to zero
* ``--beta1``: momentum term of adam
* ``--lr``: initial learning rate for adam
* ``--gan_mode``: the type of GAN objective. \[ vanilla| lsgan | wgangp \]
* ``--pool_size``: the size of image buffer that stores previously generated images
* ``--lr_policy``: learning rate policy. \[ linear | step | plateau | cosine \]
* ``--lr_decay_iters``: multiply by a gamma every ``lr_decay_iters`` iterations
* ``--weight_decay``: L2 regularization for the generator network

**Testing-specific commands**

* ``--ntest``: \# of test examples
* ``--results_dir``: saves results here (default ``./results/``)
* ``--aspect_ratio``: aspect ratio of result images
* ``--phase``: train, val, test, etc. Do not change this option to ensure proper behavior.
* ``--eval``: use eval mode during test time
* ``--num_test``: how many test images to run


