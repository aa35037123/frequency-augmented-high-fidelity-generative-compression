import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GaussianKernel(nn.Module):
    def __init__(self, in_channels=16, out_channels=16, kernel_size=9, sigma=3, groups=1):
        super(GaussianKernel, self).__init__()
        self.kernel_size = kernel_size
        self.device = 'cuda'
        self.sigma = nn.Parameter(torch.tensor([sigma], dtype=torch.float32))  # Learnable parameter
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

    def forward(self, x):
        gaussian_kernel = self._create_gaussian_kernel(self.kernel_size, self.kernel_size, self.sigma)
        # print(f'shape of kernel : {gaussian_kernel.shape}')
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0).repeat(self.out_channels, self.in_channels, 1, 1)
        # gaussian_kernel = gaussian_kernel.unsqueeze(0).repeat(self.out_channels, self.in_channels, 1, 1)
        """
        torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) 
        input : input tensor of shape (batch_size, in_channels, H, W) 
        weight : filters of the shape (out_channels, in_channels/groups, H, W)
        groups : split channel into n part
        """
        # print(f'Shape of x : {x.shape}')
        # print(f'Shape of gaussian kernel : {gaussian_kernel.shape}')
        blurred_image = nn.functional.conv2d(x, gaussian_kernel, padding=self.kernel_size//2)
        return blurred_image

    
    def _create_gaussian_kernel(self, W, H, sigma):
        # np.mgrid[start:end:step_size], end is not included ex:[-1:2:1] -> [-1, 0, 1]
        # sigma = sigma.detach().numpy()
        # x, y = np.mgrid[-(W//2):((W+1)//2), -(H//2):((H+1)//2)]
        # print(f'x : {x}')
        # print(f'y : {y}')
        x = torch.arange(-(W//2), (W+1)//2, 1)
        y = torch.arange(-(H//2), (H+1)//2, 1)
        x, y = torch.meshgrid(torch.arange(-(W//2), (W+1)//2), torch.arange(-(H//2), (H+1)//2))
        x, y = x.to(self.device), y.to(self.device)
        # x, y = np.mgrid[-(W//2):((W)//2), -(H//2):((H)//2)]
        torch_pi = torch.acos(torch.zeros(1)).item() * 2
        # torch_pi = torch.tensor(torch_pi)
        
        # print(f'torch pi : {torch_pi}')
        gaussian_kernel = 1 / (2.0 * torch_pi * sigma**2) * torch.exp(-(x**2+y**2)/(2*sigma**2)) 
        # print(f'gaussian before norm :\n{gaussian_kernel}')
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        # print(f'gaussiam kernel : {gaussian_kernel}')
        return gaussian_kernel

if __name__ == '__main__':
    # Create a simple image
    batch_size = 4
    image = torch.rand(batch_size, 16, 255, 255).to(device='cuda')

    # Create the GaussianBlur model
    kernel_size = 9
    gaussian_blur_model = GaussianKernel(kernel_size).to(device='cuda')
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(gaussian_blur_model.parameters(), lr=0.1)

    # Train the model to learn the best sigma for blurring
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        blurred_image = gaussian_blur_model(image)
        # print(f'Shape of blurred_image : {blurred_image.shape}')
        loss = criterion(blurred_image, image)  # The goal is to minimize the difference between blurred and original
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Learned Gaussian Blur Sigma:", gaussian_blur_model.sigma.item())
