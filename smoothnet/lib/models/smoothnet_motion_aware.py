# Gaoqi Li
# Impl motion aware smoothNet mentioned in the paper
# 2022/10/11


import torch
from torch import Tensor, nn
from torchinfo import summary


class SmoothNetResBlock(nn.Module):
    """Residual block module used in SmoothNet.
    Args:
        in_channels (int): Input channel number.
        hidden_channels (int): The hidden feature channel number.
        dropout (float): Dropout probability. Default: 0.5
    Shape:
        Input: (*, in_channels)
        Output: (*, in_channels)
    """

    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, in_channels)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.lrelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.lrelu(x)

        out = x + identity
        return out


class SmoothNetMotionAware(nn.Module):
    """SmoothNet is a plug-and-play temporal-only network to refine human
    poses. It works for 2d/3d/6d pose smoothing.
    "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos",
    arXiv'2021. More details can be found in the `paper
    <https://arxiv.org/abs/2112.13715>`__ .
    Note:
        N: The batch size
        T: The temporal length of the pose sequence
        C: The total pose dimension (e.g. keypoint_number * keypoint_dim)
    Args:
        window_size (int): The size of the input window.
        output_size (int): The size of the output window.
        hidden_size (int): The hidden feature dimension in the encoder,
            the decoder and between residual blocks. Default: 512
        res_hidden_size (int): The hidden feature dimension inside the
            residual blocks. Default: 256
        num_blocks (int): The number of residual blocks. Default: 3
        dropout (float): Dropout probability. Default: 0.5
    Shape:
        Input: (N, C, T) the original pose sequence
        Output: (N, C, T) the smoothed pose sequence
    """

    def __init__(self,
                 window_size: int,
                 output_size: int,
                 hidden_size: int = 512,
                 res_hidden_size: int = 256,
                 num_blocks: int = 3,
                 dropout: float = 0.5):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.res_hidden_size = res_hidden_size
        self.num_blocks = num_blocks
        self.dropout = dropout

        assert window_size >= 2, 'The window size must larger than 2'

        assert output_size <= window_size, (
            'The output size should be less than or equal to the window size.',
            f' Got output_size=={output_size} and window_size=={window_size}')

        # Build encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.LeakyReLU(0.1, inplace=True))

        self.encoder_vel = nn.Sequential(
            nn.Linear(window_size - 1, hidden_size - 1),
            nn.LeakyReLU(0.1, inplace=True))

        self.encoder_acc = nn.Sequential(
            nn.Linear(window_size - 2, hidden_size - 2),
            nn.LeakyReLU(0.1, inplace=True))

        # Build residual blocks
        res_blocks = []
        res_blocks_vel = []
        res_blocks_acc = []
        for _ in range(num_blocks):
            res_blocks.append(
                SmoothNetResBlock(
                    in_channels=hidden_size,
                    hidden_channels=res_hidden_size,
                    dropout=dropout))
            res_blocks_vel.append(
                SmoothNetResBlock(
                    in_channels=hidden_size - 1,
                    hidden_channels=res_hidden_size,
                    dropout=dropout))
            res_blocks_acc.append(
                SmoothNetResBlock(
                    in_channels=hidden_size - 2,
                    hidden_channels=res_hidden_size,
                    dropout=dropout))

        self.res_blocks = nn.Sequential(*res_blocks)
        self.res_blocks_vel = nn.Sequential(*res_blocks_vel)
        self.res_blocks_acc = nn.Sequential(*res_blocks_acc)

        self.decoder = nn.Linear(hidden_size, output_size)
        self.decoder_vel = nn.Linear(hidden_size - 1, output_size)
        self.decoder_acc = nn.Linear(hidden_size - 2, output_size)

        self.fusion_layer = nn.Linear(output_size * 3, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        N, C, T = x.shape
        x = x.to(torch.float32)

        assert T == self.window_size, (
            'Input sequence length must be equal to the window size. ',
            f'Got x.shape[2]=={T} and window_size=={self.window_size}')

        x_v = torch.diff(x, dim=2)                                                      # Vel
        x_a = torch.diff(x_v, dim=2)                                                    # Acc

        # Forward layers
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)  # [N, C, output_size]

        x_v = self.encoder_vel(x_v)
        x_v = self.res_blocks_vel(x_v)
        x_v = self.decoder_vel(x_v)

        x_a = self.encoder_acc(x_a)
        x_a = self.res_blocks_acc(x_a)
        x_a = self.decoder_acc(x_a)

        x = torch.cat([x, x_v, x_a], dim=2)

        x = self.fusion_layer(x)

        return x
