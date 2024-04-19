import torch
import torch.nn as nn
import numpy as np
from .util import init

"""CNN Modules and utils."""

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
def calculate_channel_sizes( image_channels, max_filters, num_layers):
        channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
        for i in range(1, num_layers):
            prev = channel_sizes[-1][-1]
            new = prev * 2
            channel_sizes.append((prev, new))
        return channel_sizes

def calculate_layer_size( input_size, kernel_size, stride, padding=0):
        return ((input_size - kernel_size + 2*padding) // stride) + 1



class CNNLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU, kernel_size=3, stride=1):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=hidden_size // 2,
                            kernel_size=kernel_size,
                            stride=stride)
                  ),
            active_func,
            Flatten(),
            init_(nn.Linear(hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride),
                            hidden_size)
                  ),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)), active_func)

    def forward(self, x):
        x = x / 255.0
        x = self.cnn(x)
        return x


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape):
        super(CNNBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self.hidden_size = args.hidden_size

        self.cnn = CNNLayer(obs_shape, self.hidden_size, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        x = self.cnn(x)
        return x


class MultiConvNet(nn.Module):
    def __init__(self, in_channels, num_layers, num_channels, activation, use_batchnorm=False):
        super(MultiConvNet, self).__init__()
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        
        self.conv_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()
        
        # RGB images have 3 input channels
        
        for _ in range(self.num_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, self.num_channels, kernel_size=3, padding=1))
            in_channels = self.num_channels
            
            if self.use_batchnorm:
                self.batchnorm_layers.append(nn.BatchNorm2d(self.num_channels))
        
        self.linear  = nn.Linear(self.num_channels, 1)
        self.flatten = Flatten()
        
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            
            if self.use_batchnorm:
                x = self.batchnorm_layers[i](x)
            
            x = self.activation(x)
        
        # Flatten the output before passing through the linear layer
        x = self.flatten(x)
        x = self.linear(x)
        
        return x

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, kernel_size, stride, padding, norm_type='layer', num_groups=1, nonlinearity=None ):
        """
            1. in_channels is the number of input channels to the first conv layer,
            2. out_channels is the number of output channels of the first conv layer
                and the number of input channels to the second conv layer
        """
        super(ResidualBlock, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        layers=[]
        layers.append(
                        nn.Conv2d(
                                in_channels,
                                in_channels,
                                kernel_size,
                                stride,
                                padding,
                                bias    = False)

        )
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(in_channels))
        elif norm_type == 'layer':
            layers.append(nn.GroupNorm(num_groups, in_channels))

        layers.append(nl)
        layers.append(
                        nn.Conv2d(
                                in_channels,
                                in_channels,
                                kernel_size,
                                stride,
                                padding,
                                bias    = False)

        )
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(in_channels))
        elif norm_type == 'layer':
            layers.append(nn.GroupNorm(num_groups, in_channels))

        layers.append(nl)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out =  out + x
        # each residual block doesn't wrap (res_x + x) with an activation function
        # as the next block implement ReLU as the first layer
        return out

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x
    
class Encoder(nn.Module):
    def __init__(self, 
                 in_channel, 
                 img_height,
                 img_width, 
                 hidden_dim, 
                 device, 
                 max_filters=512, 
                 num_layers=4, 
                 small_conv=False, 
                 norm_type = 'batch', 
                 num_groups=1, 
                 kernel_size=4, 
                 stride_size=2, 
                 padding_size=1, 
                 activation = nn.PReLU()
                 ):
        super(Encoder,self).__init__()
        self.nchannel    = in_channel
        self.hidden_dim  = hidden_dim
        self.img_width   = img_width
        self.img_height  = img_height
        self.device      = device
        self.enc_kernel  = kernel_size
        self.enc_stride  = stride_size
        self.enc_padding = padding_size
        self.res_kernel  = 3
        self.res_stride  = 1
        self.res_padding = 1
        self.activation  = activation
        ########################
        # ENCODER-CONVOLUTION LAYERS
        if small_conv:
            num_layers += 1
        channel_sizes = calculate_channel_sizes(
            self.nchannel, max_filters, num_layers
        )

        # Encoder
        encoder_layers = nn.ModuleList()
        # Encoder Convolutions
        for i, (in_channels, out_channels) in enumerate(channel_sizes):
            if small_conv and i == 0:
                # 1x1 Convolution
                encoder_layers.append(nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=self.enc_kernel,
                        stride=self.enc_stride,
                        padding=self.enc_padding,
                ))
                #encoder_layers.append(PrintLayer())
            else:
                encoder_layers.append( nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=self.enc_kernel,
                        stride=self.enc_stride,
                        padding=self.enc_padding,
                        bias=False,
                    ))
                #encoder_layers.append(PrintLayer())
            # Batch Norm
            if norm_type == 'batch':
                encoder_layers.append(nn.BatchNorm2d(out_channels))
            elif norm_type == 'layer':
                encoder_layers.append(nn.GroupNorm(num_groups, out_channels ))

            # ReLU
            encoder_layers.append(self.activation)

            if (i==num_layers//2):
                #add a residual Layer
                encoder_layers.append(ResidualBlock(
                        out_channels,
                        self.res_kernel,
                        self.res_stride,
                        self.res_padding,
                        norm_type=norm_type,
                        nonlinearity=self.activation
                    ))
                #encoder_layers.append(PrintLayer())

        # Flatten Encoder Output
        encoder_layers.append(nn.Flatten())

        self.encoder = nn.Sequential(*encoder_layers)
        

        # Calculate shape of the flattened image
        self.h_dim, (self.height_image_dim, self.width_image_dim) = self.get_flattened_size((self.img_height,self.img_width))

        #linear layers
        layers = []
        layers.append(nn.Linear(self.h_dim, hidden_dim, bias=False))
        if norm_type == 'batch':
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm(hidden_dim ))
        layers.append(self.activation)

        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        if norm_type == 'batch':
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm(hidden_dim ))
        layers.append(self.activation)

        self.linear_layers = nn.Sequential( *layers)
        
        self.to(device=self.device)

    def forward(self,X):
        # Encode (note ensure input tensor has the shape [batch_size, channels, height, width])    
        print(f"CNN module : check the size of input {X.shape}")
        if X.shape[1] != self.nchannel:
           X = X.permute(0, 3, 1, 2)
        print(f"CNN module permuted iput shape {X.shape}")
        h = self.encoder(X)
        print (f"size of output of encoder (CNN) {h.shape}")
        # Get latent variables
        return self.linear_layers(h)
            
    
    def get_flattened_size(self, image_dim):
        """
        image_dim is a tuple (height, width)
        """

        image_height, image_width = image_dim

        for layer in self.encoder.modules():
            if isinstance(layer, nn.Conv2d):

              kernel_size_h, kernel_size_w = layer.kernel_size
              stride_h, stride_w = layer.stride
              padding_h, padding_w = layer.padding
              filters = layer.out_channels

              image_height = calculate_layer_size(
                 image_height, kernel_size_h, stride_h, padding_h
              )
              image_width = calculate_layer_size(
                 image_width, kernel_size_w, stride_w, padding_w
              )

        return filters * image_height * image_width, (image_height, image_width)

