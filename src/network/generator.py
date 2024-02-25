import torch
import datetime
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
# Custom
from src.normalisation import channel, instance
from src.network import gaussian_kernel

class ResidualBlock(nn.Module):
    def __init__(self, input_dims, kernel_size=3, stride=1, 
                 channel_norm=True, activation='relu'):
        """
        input_dims: Dimension of input tensor (B,C,H,W)
        """
        super(ResidualBlock, self).__init__()
        
        """
        # Gaussian kernel
        """

        self.activation = getattr(F, activation)
        in_channels = input_dims[1]
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)

        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = instance.InstanceNorm2D_wrap

        pad_size = int((kernel_size-1)/2)
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.norm1 = self.interlayer_norm(in_channels, **norm_kwargs)
        self.norm2 = self.interlayer_norm(in_channels, **norm_kwargs)

    def forward(self, x):
        identity_map = x
        res = self.pad(x)
        res = self.conv1(res)
        res = self.norm1(res) 
        res = self.activation(res)

        res = self.pad(res)
        res = self.conv2(res)
        res = self.norm2(res)

        return torch.add(res, identity_map)

class Generator(nn.Module):
    def __init__(self, input_dims, batch_size, C=16, activation='relu',
                 n_residual_blocks=8, channel_norm=True, sample_noise=False,
                 noise_dim=32):

        """ 
        Generator with convolutional architecture proposed in [1].
        Upscales quantized encoder output into feature map of size C x W x H.
        Expects input size (C,16,16)
        ========
        Arguments:
        input_dims: Dimensions of quantized representation, (C,H,W)
        batch_size: Number of instances per minibatch
        C:          Encoder bottleneck depth, controls bits-per-pixel
                    C = 220 used in [1].

        [1] Mentzer et. al., "High-Fidelity Generative Image Compression", 
            arXiv:2006.09965 (2020).
        """
        
        super(Generator, self).__init__()
        
        kernel_dim = 3
        filters = [960, 480, 240, 120, 60]
        self.n_residual_blocks = n_residual_blocks
        self.sample_noise = sample_noise
        self.noise_dim = noise_dim

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=1, output_padding=1)
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        self.n_upsampling_layers = 4
        
        if channel_norm is True:
            self.interlayer_norm = channel.ChannelNorm2D_wrap
        else:
            self.interlayer_norm = instance.InstanceNorm2D_wrap

        self.pre_pad = nn.ReflectionPad2d(1)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(3)

        H0, W0 = input_dims[1:]
        heights = [2**i for i in range(5,9)]
        widths = heights
        H1, H2, H3, H4 = heights
        W1, W2, W3, W4 = widths 


        # (16,16) -> (16,16), with implicit padding
        self.conv_block_init = nn.Sequential(
            self.interlayer_norm(C, **norm_kwargs),
            self.pre_pad,
            nn.Conv2d(C, filters[0], kernel_size=(3,3), stride=1),
            self.interlayer_norm(filters[0], **norm_kwargs),
        )

        if sample_noise is True:
            # Concat noise with latent representation
            filters[0] += self.noise_dim

        for m in range(n_residual_blocks):
            resblock_m = ResidualBlock(input_dims=(batch_size, filters[0], H0, W0), 
                channel_norm=channel_norm, activation=activation)
            self.add_module(f'resblock_{str(m)}', resblock_m)
        
        # (16,16) -> (32,32)
        self.upconv_block1 = nn.Sequential(
            nn.ConvTranspose2d(filters[0], filters[1], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[1], **norm_kwargs),
            self.activation(),
        )

        self.upconv_block2 = nn.Sequential(
            nn.ConvTranspose2d(filters[1], filters[2], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[2], **norm_kwargs),
            self.activation(),
        )

        self.upconv_block3 = nn.Sequential(
            nn.ConvTranspose2d(filters[2], filters[3], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[3], **norm_kwargs),
            self.activation(),
        )

        self.upconv_block4 = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[4], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[4], **norm_kwargs),
            self.activation(),
        )

        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(filters[-1], 3, kernel_size=(7,7), stride=1),
        )

        """
        # Gaussian kernel
        """
        self.gaussian_kernel_size = 9
        self.sigma_init = 3 
        self.gaussian_kernel_for_fcm2 = gaussian_kernel.GaussianKernel(in_channels=filters[3], out_channels=filters[3],
            sigma=self.sigma_init, kernel_size=self.gaussian_kernel_size)
        self.gaussian_kernel_for_fcm1 = gaussian_kernel.GaussianKernel(in_channels=filters[4], out_channels=filters[4],
            sigma=self.sigma_init, kernel_size=self.gaussian_kernel_size)
        """
        # Frequency Complement Module
        """
        self.FCM2 = nn.Sequential(
            nn.LayerNorm((128, 128), eps=1e-05, elementwise_affine=True, device=None, dtype=None),
            nn.ReLU(), 
            nn.Dropout(p=0.5),
            nn.Conv2d(filters[3], 64, kernel_size=(3,3), stride=1, padding='same'),
            nn.LayerNorm((128, 128), eps=1e-05, elementwise_affine=True, device=None, dtype=None),
            nn.ReLU(), 
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding='same'),
            nn.LayerNorm((128, 128), eps=1e-05, elementwise_affine=True, device=None, dtype=None),
            nn.ReLU(), 
            nn.Dropout(p=0.5),
            nn.Conv2d(64, filters[3], kernel_size=(3,3), stride=1, padding='same'),
        )

        self.FCM1 = nn.Sequential(
            nn.LayerNorm((256, 256), eps=1e-05, elementwise_affine=True, device=None, dtype=None),
            nn.ReLU(), 
            nn.Dropout(p=0.5),
            nn.Conv2d(filters[4], 64, kernel_size=(3,3), stride=1, padding='same'),
            nn.LayerNorm((256, 256), eps=1e-05, elementwise_affine=True, device=None, dtype=None),
            nn.ReLU(), 
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding='same'),
            nn.LayerNorm((256, 256), eps=1e-05, elementwise_affine=True, device=None, dtype=None),
            nn.ReLU(), 
            nn.Dropout(p=0.5),
            nn.Conv2d(64, filters[4], kernel_size=(3,3), stride=1, padding='same'),
        )
        
        self.activate_path = '/home/pc3424/DL_final/high-fidelity-generative-compression/experiments/test_activate/compression_fix_pretrain_high/decoder'
        if not os.path.exists(self.activate_path):
            os.makedirs(self.activate_path)

    def forward(self, x):
        
        head = self.conv_block_init(x)

        if self.sample_noise is True:
            B, C, H, W = tuple(head.size())
            z = torch.randn((B, self.noise_dim, H, W)).to(head)
            head = torch.cat((head,z), dim=1)

        for m in range(self.n_residual_blocks):
            resblock_m = getattr(self, f'resblock_{str(m)}')
            if m == 0:
                x = resblock_m(head)
            else:
                x = resblock_m(x)
        
        x += head
        # B block 
        x = self.upconv_block1(x)
        x = self.upconv_block2(x)
        B2 = self.upconv_block3(x)
        # print(B2.shape)
        # decoder2s
        C2 = self.FCM2(B2)
        # save_image(self.tensor_reshape(C2), os.path.join(self.activate_path, 'C2_{:%Y_%m_%d_%H:%M}.jpg'.format(datetime.datetime.now())))
        time_signature = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()).replace(':', '_')
        fname = '{}_{}_{}'.format('openimages', 'compression_gan', time_signature)
        figures_save = os.path.join('experiments', fname, 'figures')
        # save_image(C2, os.path.join(figures_save, '_C2.jpg'))
        C2_gaussian = self.gaussian_kernel_for_fcm2(C2)
        # print(f'sigma of C2_gaussian : {self.gaussian_kernel_for_fcm2.sigma.item()}')
        x = B2 + C2
        
        B1 = self.upconv_block4(x)
        # print(B1.shape)
        # decoder1
        C1 = self.FCM1(B1)
        # save_image(self.tensor_reshape(C1), os.path.join(self.activate_path, 'C1_{:%Y_%m_%d_%H:%M}.jpg'.format(datetime.datetime.now())))
        time_signature = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()).replace(':', '_')
        fname = '{}_{}_{}'.format('openimages', 'compression_gan', time_signature)
        figures_save = os.path.join('experiments', fname, 'figures')
        # save_image(C1, os.path.join(figures_save, '_C1.jpg'))
        C1_gaussian = self.gaussian_kernel_for_fcm1(C1)
        # print(f'sigma of C1_gaussian : {self.gaussian_kernel_for_fcm1.sigma.item()}')
        x = B1 + C1
        
        out = self.conv_block_out(x)
        
        return out, C1_gaussian, C2_gaussian
    
    def tensor_reshape(self, tensor):
        shape = tensor.shape
        tensor_reshape = tensor.permute(1, 0, 2, 3).reshape(shape[1] * shape[0], 1, shape[2], shape[3])
        tensor_reshape = torch.clamp(tensor_reshape, 0, 1)
        tensor_reshape = tensor_reshape.to(torch.float32)
        return tensor_reshape


if __name__ == "__main__":

    C = 8
    y = torch.randn([3,C,16,16])
    y_dims = y.size()
    G = Generator(y_dims[1:], y_dims[0], C=C, n_residual_blocks=3, sample_noise=True)

    x_hat = G(y)
    print(x_hat.size())