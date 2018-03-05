import torch
from torch import nn
from torch.nn import functional as F


def custom_conv2d(conv2d):
    def decorated(input_features, output_features, kernel_size, stride = 1, padding = 0, *args, **kwargs):
        def convert(image_size):
            return (image_size + 2*padding - kernel_size)//stride + 1
        return conv2d(input_features, output_features, kernel_size, stride, padding, *args, **kwargs), convert
    return decorated

conv2d = custom_conv2d(torch.nn.Conv2d)

class SimpleNet(torch.nn.Module):
    """Simple convolutional networ: 2 conv layers followed by 2 fc layers.

     model = SimpleNet(# input channels, # num of output classes, image_size)
     model(data) performs the forward computation
    """

    def __init__(self, input_features, output_classes, image_size):
        super(SimpleNet, self).__init__()
        self.conv0, f_size = conv2d(input_features, 20, kernel_size = 5, stride = 1, padding = 5 // 2)
        image_size = f_size(image_size)
        self.conv1, f_size = conv2d(20, 20, kernel_size = 5, stride = 2, padding = 0)
        image_size = f_size(image_size)
        self.conv2, f_size = conv2d(20, 20, kernel_size = 5, stride = 2, padding = 0)
        image_size = f_size(image_size)
        self.fc1 = torch.nn.Linear(20*image_size**2, 60)
        self.fc2 = torch.nn.Linear(60, output_classes)

    def forward(self, x):
        x = F.relu( self.conv0(x) )
        x = F.relu( self.conv1(x) )
        x = F.relu( self.conv2(x) )
        x = x.view(x.size(0), -1)
        x = F.relu( self.fc1(x) )
        x = self.fc2(x)
        x = F.log_softmax(x, dim = 1)
        return x
