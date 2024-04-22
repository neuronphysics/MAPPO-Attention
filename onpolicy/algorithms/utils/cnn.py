import torch
import torch.nn as nn
import numpy as np
from .util import init, calculate_conv_params

"""CNN Modules and utils."""


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


def calculate_channel_sizes(image_channels, max_filters, num_layers):
    channel_sizes = [(image_channels, max_filters // np.power(2, num_layers - 1))]
    for i in range(1, num_layers):
        prev = channel_sizes[-1][-1]
        new = prev * 2
        channel_sizes.append((prev, new))
    return channel_sizes


def calculate_layer_size(input_size, kernel_size, stride, padding=0):
    return ((input_size - kernel_size + 2 * padding) // stride) + 1


class CNNLayer(nn.Module):
    def __init__(self, obs_shape, hidden_size, use_orthogonal, use_ReLU):
        super(CNNLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        if obs_shape[0] == 3:
            input_channel = obs_shape[0]
            input_width = obs_shape[1]
            input_height = obs_shape[2]
        elif obs_shape[2] == 3:
            input_channel = obs_shape[2]
            input_width = obs_shape[0]
            input_height = obs_shape[1]

        kernel_size, stride, padding = calculate_conv_params((input_width, input_height, input_channel))

        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=hidden_size // 2,
                            kernel_size=kernel_size,
                            stride=stride)),
            nn.BatchNorm2d(hidden_size // 2),
            active_func,
            Flatten(),
            init_(nn.Linear(
                hidden_size // 2 * (input_width - kernel_size + stride) * (input_height - kernel_size + stride),
                hidden_size)),
            nn.LayerNorm(hidden_size),
            active_func,
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            active_func)

    def forward(self, x):
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)  # Rearrange the dimensions
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


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, kernel_size, stride, padding, image_height, image_width, norm_type='layer',
                 num_groups=1, nonlinearity=None):
        """
            1. in_channels is the number of input channels to the first conv layer,
            2. out_channels is the number of output channels of the first conv layer
                and the number of input channels to the second conv layer
        """
        super(ResidualBlock, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        self.activation = nl
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                bias=False)

        )
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(in_channels))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm([in_channels, image_height, image_width]))
        else:
            layers.append(nn.GroupNorm(num_groups, in_channels))

        layers.append(nl)
        layers.append(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                bias=False)

        )
        if norm_type == 'batch':
            layers.append(nn.BatchNorm2d(in_channels))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm([in_channels, image_height, image_width]))
        else:
            layers.append(nn.GroupNorm(num_groups, in_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = out + x
        # each residual block doesn't wrap (res_x + x) with an activation function
        # as the next block implement ReLU as the first layer
        return self.activation(out)


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    """
    Generate an array of position indices for an N-D input array.
    Args:
      index_dims (`List[int]`):
        The shape of the index dimensions of the input array.
      output_range (`Tuple[float]`, *optional*, defaults to `(-1.0, 1.0)`):
        The min and max values taken by each input index dimension.
    Returns:
      `torch.FloatTensor` of shape `(index_dims[0], index_dims[1], .., index_dims[-1], N)`.
    """

    def _linspace(n_xels_per_dim):
        return torch.linspace(start=output_range[0], end=output_range[1], steps=n_xels_per_dim, dtype=torch.float32)

    dim_ranges = [_linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]

    array_index_grid = torch.meshgrid(*dim_ranges)

    return torch.stack(array_index_grid, dim=-1)


def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    """
    Checks or builds spatial position features (x, y, ...).
    Args:
      pos (`torch.FloatTensor`):
        None, or an array of position features. If None, position features are built. Otherwise, their size is checked.
      index_dims (`List[int]`):
        An iterable giving the spatial/index size of the data to be featurized.
      batch_size (`int`):
        The batch size of the data to be featurized.
    Returns:
        `torch.FloatTensor` of shape `(batch_size, prod(index_dims))` an array of position features.
    """
    if pos is None:
        pos = build_linear_positions(index_dims)
        pos = torch.broadcast_to(pos[None], (batch_size,) + pos.shape)
        pos = torch.reshape(pos, [batch_size, np.prod(index_dims), -1])

    else:
        # Just a warning label: you probably don't want your spatial features to
        # have a different spatial layout than your pos coordinate system.
        # But feel free to override if you think it'll work!
        if pos.shape[-1] != len(index_dims):
            raise ValueError("Spatial features have the wrong number of dimensions.")
    return pos


def generate_fourier_features(pos, num_bands, max_resolution=(224, 224), concat_pos=True, sine_only=False):
    """
    Generate a Fourier frequency position encoding with linear spacing.
    Args:
      pos (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`):
        The Tensor containing the position of n points in d dimensional space.
      num_bands (`int`):
        The number of frequency bands (K) to use.
      max_resolution (`Tuple[int]`, *optional*, defaults to (224, 224)):
        The maximum resolution (i.e. the number of pixels per dim). A tuple representing resolution for each dimension.
      concat_pos (`bool`, *optional*, defaults to `True`):
        Whether to concatenate the input position encoding to the Fourier features.
      sine_only (`bool`, *optional*, defaults to `False`):
        Whether to use a single phase (sin) or two (sin/cos) for each frequency band.
    Returns:
      `torch.FloatTensor` of shape `(batch_size, sequence_length, n_channels)`: The Fourier position embeddings. If
      `concat_pos` is `True` and `sine_only` is `False`, output dimensions are ordered as: [dim_1, dim_2, ..., dim_d,
      sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ..., sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d), cos(pi*f_1*dim_1),
      ..., cos(pi*f_K*dim_1), ..., cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)], where dim_i is pos[:, i] and f_k is the
      kth frequency band.
    """

    batch_size = pos.shape[0]

    min_freq = 1.0
    # Nyquist frequency at the target resolution:
    freq_bands = torch.stack(
        [torch.linspace(start=min_freq, end=res / 2, steps=num_bands) for res in max_resolution], dim=0
    )

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos[0, :, :][:, :, None] * freq_bands[None, :, :]
    per_pos_features = torch.reshape(per_pos_features, [-1, np.prod(per_pos_features.shape[1:])])

    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(np.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1
        )

    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat([pos, per_pos_features.expand(batch_size, -1, -1)], dim=-1)

    

    return per_pos_features


class FourierPositionEncoding(nn.Module):
    """Fourier (Sinusoidal) position encoding."""

    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        super().__init__()
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    @property
    def num_dimensions(self) -> int:
        return len(self.max_resolution)

    def output_size(self):
        """Returns size of positional encodings last dimension."""
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += self.num_dimensions

        return encoding_size

    def forward(self, index_dims, batch_size, device, pos=None):

        pos = _check_or_build_spatial_positions(pos, index_dims, batch_size)
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        ).to(device)
        return fourier_pos_enc


class TrainablePositionEncoding(nn.Module):
    """Trainable position encoding."""

    def __init__(self, index_dims, num_channels=128):
        super().__init__()
        self._num_channels = num_channels
        self._index_dims = index_dims
        index_dim = np.prod(index_dims)
        self.position_embeddings = nn.Parameter(torch.randn(index_dim, num_channels))

    @property
    def num_dimensions(self) -> int:
        if isinstance(self._index_dims, int):
            return 1
        return len(self._index_dims)

    def output_size(self, *args, **kwargs) -> int:
        return self._num_channels

    def forward(self, batch_size):
        position_embeddings = self.position_embeddings

        if batch_size is not None:
            position_embeddings = position_embeddings.expand(batch_size, -1, -1)
        return position_embeddings


def build_position_encoding(
        position_encoding_type,
        out_channels=None,
        project_pos_dim=-1,
        trainable_position_encoding_kwargs=None,
        fourier_position_encoding_kwargs=None,
):
    """
    Builds the position encoding.
    Args:
    - out_channels: refers to the number of channels of the position encodings.
    - project_pos_dim: if specified, will project the position encodings to this dimension.
    """

    if position_encoding_type == "trainable":
        if not trainable_position_encoding_kwargs:
            raise ValueError("Make sure to pass trainable_position_encoding_kwargs")
        output_pos_enc = TrainablePositionEncoding(**trainable_position_encoding_kwargs)
    elif position_encoding_type == "fourier":
        # We don't use the index_dims argument, as this is only known during the forward pass
        if not fourier_position_encoding_kwargs:
            raise ValueError("Make sure to pass fourier_position_encoding_kwargs")
        output_pos_enc = FourierPositionEncoding(**fourier_position_encoding_kwargs)
    else:
        raise ValueError(f"Unknown position encoding type: {position_encoding_type}.")

    # Optionally, project the position encoding to a target dimension:
    positions_projection = nn.Linear(out_channels, project_pos_dim) if project_pos_dim > 0 else nn.Identity()

    return output_pos_enc, positions_projection


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
                 norm_type='layer',
                 num_groups=1,
                 kernel_size=4,
                 stride_size=2,
                 padding_size=1,
                 activation=nn.ReLU(),
                 num_bands_positional_encoding=10,
                 project_pos_dim=-1,
                 position_encoding_type: str = "trainable",
                 concat_or_add_pos: str = "concat",
                 ):
        super(Encoder, self).__init__()
        self.nchannel = in_channel
        self.hidden_dim = hidden_dim
        self.img_width = img_width
        self.img_height = img_height
        self.device = device
        self.enc_kernel = kernel_size
        self.enc_stride = stride_size
        self.enc_padding = padding_size
        self.res_kernel = 3
        self.res_stride = 1
        self.res_padding = 1
        self.activation = activation

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(f"Invalid value {concat_or_add_pos} for concat_or_add_pos.")
        self.position_encoding_type = position_encoding_type
        self.concat_or_add_pos = concat_or_add_pos
        self.project_pos_dim = project_pos_dim
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
                # encoder_layers.append(PrintLayer())
            else:
                encoder_layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=self.enc_kernel,
                    stride=self.enc_stride,
                    padding=self.enc_padding,
                    bias=False,
                ))
                # encoder_layers.append(PrintLayer())
            # Batch Norm
            if norm_type == 'batch':
                encoder_layers.append(nn.BatchNorm2d(out_channels))
            elif norm_type == 'layer':
                encoder_layers.append(nn.LayerNorm([out_channels, img_height, img_width]))
            else:
                encoder_layers.append(nn.GroupNorm(num_groups, out_channels))

            # ReLU
            encoder_layers.append(self.activation)

            if (i == num_layers // 2):
                # add a residual Layer
                encoder_layers.append(ResidualBlock(
                    out_channels,
                    self.res_kernel,
                    self.res_stride,
                    self.res_padding,
                    image_height=img_height,
                    image_width=img_width,
                    norm_type=norm_type,
                    nonlinearity=self.activation
                ))
                # encoder_layers.append(PrintLayer())

        self.encoder = nn.Sequential(*encoder_layers)
        self.out_channels = channel_sizes[-1][1]
        # Calculate shape of the flattened image
        self.h_dim, (self.height_image_dim, self.width_image_dim) = self.get_flattened_size(
            (self.img_height, self.img_width))

        # Position embeddings
        position_encoding_kwargs = dict(
            concat_pos=True, max_resolution=(self.img_height * self.out_channels // 2, self.img_width
                                             * self.out_channels // 2),
            num_bands=num_bands_positional_encoding, sine_only=False
        )
        trainable_position_encoding_kwargs = dict(num_channels=self.out_channels, index_dims=1)
        self.position_embeddings, self.positions_projection = build_position_encoding(
            position_encoding_type=position_encoding_type,
            out_channels=self.out_channels,
            project_pos_dim=self.project_pos_dim,
            trainable_position_encoding_kwargs=trainable_position_encoding_kwargs,
            fourier_position_encoding_kwargs=position_encoding_kwargs,
        )

        # Adjust the input size of the first linear layer to account for the positional features

        # linear layers
        layers = []
        # Flatten Encoder Output
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.out_channels * (1 + self.img_height * self.img_width), hidden_dim, bias=False))
        if norm_type == 'batch':
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(self.activation)

        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        if norm_type == 'batch':
            layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm_type == 'layer':
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(self.activation)
        self.linear_layers = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.to(device=self.device)

    def normalize(self, X):
        # Example normalization: scale to [0, 1]
        return X / 255.0

    @property
    def num_channels(self) -> int:
        # Let's assume that the number of resolutions (in the context of image preprocessing)
        # of the input data is 2 or 3 depending on whether we are processing image or video respectively.
        # In this case, for convenience, we will declare is_temporal variable,
        # which will show whether the data has a temporal dimension or not.
        is_temporal = self.position_embeddings.num_dimensions > 2

        # position embedding
        if self.project_pos_dim > 0:
            pos_dim = self.project_pos_dim
        else:
            pos_dim = self.position_embeddings.output_size()
        if self.concat_or_add_pos == "add":
            return pos_dim

        # inputs
        inp_dim = self.out_channels

        return inp_dim + pos_dim
    
    def _init_weights(self, m):
        import math
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()


    def _build_network_inputs(self, inputs: torch.Tensor, network_input_is_1d: bool = True):
        """
        Construct the final input, including position encoding.
        This method expects the inputs to always have channels as last dimension.
        """
        batch_size = inputs.shape[0]

        index_dims = inputs.shape[1:-1]
        indices = np.prod(index_dims).item()
        
        # Flatten input features to a 1D index dimension if necessary.
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = torch.reshape(inputs, [batch_size, indices, -1])

        # Construct the position encoding.
        if self.position_encoding_type == "trainable":
            pos_enc = self.position_embeddings(batch_size)
        elif self.position_encoding_type == "fourier":
            
            pos_enc = self.position_embeddings(index_dims, batch_size, device=inputs.device)

        # Optionally project them to a target dimension.
        pos_enc = self.positions_projection(pos_enc)
        
        if not network_input_is_1d:
            # Reshape pos to match the input feature shape
            # if the network takes non-1D inputs
            sh = inputs.shape
            pos_enc = torch.reshape(pos_enc, list(sh)[:-1] + [-1])
        if self.concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=1)
            
        elif self.concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc
        return inputs_with_pos, inputs

    def forward(self, X: torch.Tensor, network_input_is_1d: bool = True):
        # Encode (note ensure input tensor has the shape [batch_size, channels, height, width])    
        
        X = self.normalize(X)
        if X.shape[1] != self.nchannel:
            X = X.permute(0, 3, 1, 2)
        
        inputs = self.encoder(X)
        
        if inputs.ndim == 4:
            # move channels to last dimension, as the _build_network_inputs method below expects this
            inputs = torch.permute(inputs, (0, 2, 3, 1))
        # Get latent variables
        
        inputs, inputs_without_pos = self._build_network_inputs(inputs, network_input_is_1d)
        return self.linear_layers(inputs)

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


class ResidualBlock_deconv(nn.Module):
    def __init__(self, channel, kernel_size, stride, padding, norm_type="layer", num_groups=1, nonlinearity=None):
        super(ResidualBlock_deconv, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        self.conv1 = nn.ConvTranspose2d(channel, channel, kernel_size, stride, padding)
        if norm_type == "batch":
            self.norm1 = nn.BatchNorm2d(channel)
        elif norm_type == "layer":
            self.norm1 = nn.LayerNorm(channel)
        else:
            self.norm1 = nn.GroupNorm(num_groups, channel)
        self.activation = nl
        self.conv2 = nn.ConvTranspose2d(channel, channel, kernel_size, stride, padding)
        if norm_type == "batch":
            self.norm2 = nn.BatchNorm2d(channel)
        elif norm_type == "layer":
            self.norm2 = nn.LayerNorm(channel)
        else:
            self.norm2 = nn.GroupNorm(num_groups, channel)

    def forward(self, x):
        res = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.activation(self.norm2(self.conv2(out)))
        out = out + res
        return out


class Decoder(nn.Module):

    def __init__(self,
                 in_channel,
                 hidden_dim,
                 extend_dim,
                 image_height,
                 image_width,
                 max_filters=512,
                 num_layers=4,
                 small_conv=False,
                 norm_type='batch',
                 num_groups=1,
                 kernel_size=4,
                 stride_size=2,
                 padding_size=1,
                 activation=nn.GELU()):
        super(Decoder, self).__init__()

        self.nchannel = in_channel
        self.hidden_dim = hidden_dim
        self.img_width = image_width
        self.dec_kernel = kernel_size
        self.dec_stride = stride_size
        self.dec_padding = padding_size
        self.res_kernel = 3
        self.res_stride = 1
        self.res_padding = 1
        self.activation = activation

        if small_conv:
            num_layers += 1
        channel_sizes = calculate_channel_sizes(
            self.nchannel, max_filters, num_layers
        )

        # Decoder
        decoder_layers = nn.ModuleList()
        # Feedforward/Dense Layer to expand our latent dimensions
        decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))

        if norm_type == 'batch':
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
        elif norm_type == 'layer':
            decoder_layers.append(nn.LayerNorm(hidden_dim))

        decoder_layers.append(self.activation)
        decoder_layers.append(torch.nn.Linear(hidden_dim, extend_dim, bias=False))
        if norm_type == 'batch':
            decoder_layers.append(nn.BatchNorm1d(extend_dim))
        elif norm_type == 'layer':
            decoder_layers.append(nn.LayerNorm(extend_dim))

        decoder_layers.append(self.activation)
        # Unflatten to a shape of (Channels, Height, Width)
        decoder_layers.append(
            nn.Unflatten(1, (int(extend_dim / (image_height * image_width)), image_height, image_width)))
        # Decoder Convolutions

        for i, (out_channels, in_channels) in enumerate(channel_sizes[::-1]):
            if small_conv and i == num_layers - 1:
                # 1x1 Transposed Convolution
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=self.dec_kernel,
                        stride=self.dec_stride,
                        padding=self.dec_padding,
                    )
                )
            else:
                # Add Transposed Convolutional Layer
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=self.dec_kernel,
                        stride=self.dec_stride,
                        padding=self.dec_padding,
                        bias=False,
                    )
                )
            if norm_type == 'batch':
                decoder_layers.append(nn.BatchNorm2d(out_channels))
            elif norm_type == 'layer':
                decoder_layers.append(nn.GroupNorm(num_groups, out_channels))

            # ReLU if not final layer
            if i != num_layers - 1:
                decoder_layers.append(self.activation)
            # Sigmoid if final layer
            else:
                decoder_layers.append(nn.Sigmoid())
            if (i == num_layers // 2):
                # add a residual Layer
                decoder_layers.append(
                    ResidualBlock_deconv(
                        out_channels,
                        self.res_kernel,
                        self.res_stride,
                        self.res_padding,
                        nonlinearity=self.activation
                    )
                )

        self.decoder = nn.Sequential(*decoder_layers)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(device=self.device)
        

    def forward(self, x):
        return self.decoder(x)
