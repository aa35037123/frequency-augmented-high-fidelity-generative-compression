import torch
import os
from torchvision.utils import save_image
import datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.normalisation import channel, instance
from src.network import gaussian_kernel

class Encoder(nn.Module):
    def __init__(self, image_dims, batch_size, activation='relu', C=220,
                 channel_norm=True):
        """ 
        Encoder with convolutional architecture proposed in [1].
        Projects image x ([C_in,256,256]) into a feature map of size C x W/16 x H/16
        ========
        Arguments:
        image_dims:  Dimensions of input image, (C_in,H,W)
        batch_size:  Number of instances per minibatch
        C:           Bottleneck depth, controls bits-per-pixel
                     C = {2,4,8,16}

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        # input img : (8 x 3 x 256 x 256)
        # C is the final output channel number
        super(Encoder, self).__init__()
        
        kernel_dim = 3
        filters = (60, 120, 240, 480, 960)

        # Images downscaled to 500 x 1000 + randomly cropped to 256 x 256
        im_channels = image_dims[0]
        # assert image_dims == (im_channels, 256, 256), 'Crop image to 256 x 256!'

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=0, padding_mode='reflect')
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        self.n_downsampling_layers = 4
        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = instance.InstanceNorm2D_wrap

        self.pre_pad = nn.ReflectionPad2d(3)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(1)

        heights = [2**i for i in range(4,9)][::-1]
        widths = heights
        H1, H2, H3, H4, H5 = heights
        W1, W2, W3, W4, W5 = widths 
        
        """
        # Gaussian kernel
        """
        self.gaussian_kernel_size = 9
        self.sigma_init = 3 
        self.gaussian_kernel_for_block1 = gaussian_kernel.GaussianKernel(in_channels=filters[0], out_channels=filters[0],
            sigma=self.sigma_init, kernel_size=self.gaussian_kernel_size)
        self.gaussian_kernel_for_block2 = gaussian_kernel.GaussianKernel(in_channels=filters[1], out_channels=filters[1],
            sigma=self.sigma_init, kernel_size=self.gaussian_kernel_size)
        
        # (256,256) -> (256,256), with implicit padding
        self.conv_block1 = nn.Sequential(
            self.pre_pad,
            nn.Conv2d(im_channels, filters[0], kernel_size=(7,7), stride=1),
            self.interlayer_norm(filters[0], **norm_kwargs),
            self.activation(),
        )

        # (256,256) -> (128,128)
        self.conv_block2 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[1], **norm_kwargs),
            self.activation(),
        )

        # (128,128) -> (64,64)
        self.conv_block3 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[2], **norm_kwargs),
            self.activation(),
        )

        # (64,64) -> (32,32)
        self.conv_block4 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[3], **norm_kwargs),
            self.activation(),
        )

        # (32,32) -> (16,16)
        self.conv_block5 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[3], filters[4], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[4], **norm_kwargs),
            self.activation(),
        )
        
        # Project channels onto space w/ dimension C
        # Feature maps have dimension C x W/16 x H/16
        # (16,16) -> (16,16)
        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(filters[4], C, kernel_dim, stride=1),
        )
        self.activate_path = '/home/pc3424/DL_final/high-fidelity-generative-compression/experiments/test_activate/compression_fix_pretrain_high/encoder'
        if not os.path.exists(self.activate_path):
            os.makedirs(self.activate_path)
                
    def forward(self, x):
        # print(f'in encoder stage, input : {x.shape}')
        x = self.conv_block1(x)
        A1 = x
        # print(f'Shape of A1 : {A1.shape}')
        # print(f'A1: {A1}')
        # save_image(self.tensor_reshape(A1), os.path.join(self.activate_path, 'A1_{:%Y_%m_%d_%H:%M}.jpg'.format(datetime.datetime.now())))
        A1_gaussian = self.gaussian_kernel_for_block1(A1)
        # print(f'sigma of A1_gaussian : {self.gaussian_kernel_for_block1.sigma.item()}')
        # encoder1
        x = self.conv_block2(x)
        A2 = x
        # save_image(self.tensor_reshape(A2), os.path.join(self.activate_path, 'A2_{:%Y_%m_%d_%H:%M}.jpg'.format(datetime.datetime.now())))
        time_signature = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()).replace(':', '_')
        fname = '{}_{}_{}'.format('openimages', 'compression_gan', time_signature)
        figures_save = os.path.join('experiments', fname, 'figures')
        # save_image(A2, os.path.join(figures_save, '_A2.jpg'))
        A2_gaussian = self.gaussian_kernel_for_block2(A2)
        # print(f'sigma of A2_gaussian : {self.gaussian_kernel_for_block2.sigma.item()}')
        # encoder2
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        out = self.conv_block_out(x)
        return out, A1_gaussian, A2_gaussian

    def tensor_reshape(self, tensor):
        shape = tensor.shape
        tensor_reshape = tensor.permute(1, 0, 2, 3).reshape(shape[1] * shape[0], 1, shape[2], shape[3])
        tensor_reshape = torch.clamp(tensor_reshape, 0, 1)
        tensor_reshape = tensor_reshape.to(torch.float32)
        return tensor_reshape
if __name__ == "__main__":
    B = 2
    C = 7
    print('Image 1')
    x = torch.randn((B,3,256,256))
    x_dims = tuple(x.size())
    E = Encoder(image_dims=x_dims[1:], batch_size=B, C=C)

