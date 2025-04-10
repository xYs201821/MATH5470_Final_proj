import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CNN20(nn.Module):
    def __init__(self,
                drop_out=0.5,
                layers=3,
                batch_normalization=True,
                xavier_initialization=True,
                conv_filters=64,
                kernel_sizes=[5, 3],
                strides=[3, 1],
                dilation=[2, 1],
                activation="LReLu"):
        super(CNN20, self).__init__()
        
        # Convert lists to tuples for kernel sizes, strides, and dilation
        self.kernel_sizes = tuple(kernel_sizes) if isinstance(kernel_sizes, list) else kernel_sizes
        self.strides = tuple(strides) if isinstance(strides, list) else strides
        self.dilation = tuple(dilation) if isinstance(dilation, list) else dilation
        
        # Store other parameters
        self.drop_out = drop_out
        self.activation_type = activation.lower()
        self.conv_filters = conv_filters
        self.layers = layers
        self.batch_normalization = batch_normalization
        self.xavier_initialization = xavier_initialization
        
        # Calculate number of filters for each layer
        self.filters = [1] + [conv_filters * (2 ** i) for i in range(layers)]
        
        # Create activation function
        self.activation = self._get_activation()
        
        # Create convolutional blocks
        self.blocks = nn.ModuleList()
        dummy_input = torch.randn(1, 1, 64, 60)
        for i in range(layers):
            if i == 0:
                padding_h = self._calculate_padding(
                    kernel_size=self.kernel_sizes[0], 
                    stride=self.strides[0],           
                    dilation=self.dilation[0]         
                )
                padding_w = self._calculate_padding(
                    kernel_size=self.kernel_sizes[1],  
                    stride=self.strides[1],          
                    dilation=self.dilation[1]        
                )
            else:
                padding_h = (self.kernel_sizes[0] - 1) // 2
                padding_w = (self.kernel_sizes[1] - 1) // 2

            in_channels = self.filters[i]
            out_channels = self.filters[i+1]

            block = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.kernel_sizes,
                    stride=self.strides if i == 0 else (1, 1),
                    dilation=self.dilation if i == 0 else (1, 1),
                    padding=(padding_h, padding_w)
                ),
                nn.BatchNorm2d(out_channels) if self.batch_normalization else nn.Identity(),
                self.activation,
                nn.MaxPool2d(
                    kernel_size=(2, 1),
                    ceil_mode=True
                )
            )
            dummy_input = block(dummy_input)
            self.blocks.append(block)
        fc_input_dim = dummy_input.size(1) * dummy_input.size(2) * dummy_input.size(3)
        self.fc = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(fc_input_dim, 2)
        )
        self.apply(self._init_weights)
    
    def _calculate_output_dim(self, input_dim, kernel_size, stride, dilation):
        return (input_dim - kernel_size + 2 * self._calculate_padding(kernel_size, stride, dilation)) // stride
    
    def _calculate_padding(self, kernel_size, stride, dilation):
        return ((stride - 1) + dilation * (kernel_size - 1)) // 2
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.fc(x.view(x.size(0), -1))
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight) if self.xavier_initialization else init.eye_(m.weight)
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight) if self.xavier_initialization else init.eye_(m.weight)
            init.zeros_(m.bias)
    
    def _get_activation(self):
        activation_map ={
            "lrelu": nn.LeakyReLU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "celu": nn.CELU(),
            "gelu": nn.GELU(),
        }
        if self.activation_type not in activation_map:
            raise ValueError(f"Activation type {self.activation_type} not supported")
        return activation_map[self.activation_type]
