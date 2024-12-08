import torch
import torch.nn.functional as F
import torch.nn as nn

LRELU_SLOPE = 0.1


class ResBlock(nn.Module):
    def __init__(self, kernel_size, channels, D_r): # D_r[n]
        super().__init__()

        self.convs = nn.ModuleList()

        for m in range(len(D_r)):
            layer = nn.ModuleList()
            for l in range(len(D_r[m])):
                layer.append(nn.Sequential(
                    nn.LeakyReLU(LRELU_SLOPE),
                    nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, dilation=D_r[m][l], padding="same")
                ))
            self.convs.append(layer)

    def forward(self, x):
        for conv in self.convs:
            for layer in conv:
                y = layer(x)
            x = x + y
        return x


class MRF(nn.Module):
    def __init__(self, channels, k_r, D_r):
        super().__init__()

        self.res_blocks = nn.ModuleList()

        for n in range(len(k_r)):
            self.res_blocks.append(ResBlock(k_r[n], channels, D_r[n]))

    def forward(self, x):
        res = None
        for res_block in self.res_blocks:
            if res is None:
                res = res_block(x)
            else:
                res = res + res_block(x)
        return res / len(self.res_blocks)


class Generator(nn.Module):
    def __init__(self, h_u, k_u, k_r, D_r):
        super().__init__()

        self.input_conv = nn.Conv1d(in_channels=80, out_channels=h_u, kernel_size=7, dilation=1, padding="same")
        
        self.t_convs = nn.ModuleList()

        for i in range(len(k_u)):
            self.t_convs.append(nn.Sequential(
                nn.LeakyReLU(LRELU_SLOPE),
                nn.ConvTranspose1d(in_channels=h_u // 2**i, 
                                   out_channels=h_u // 2**(i + 1), 
                                   kernel_size=k_u[i], 
                                   stride=k_u[i] // 2, 
                                   padding=k_u[i] // 2 - k_u[i] // 4
                                   ),
                MRF(h_u // 2**(i + 1), k_r, D_r)
            ))

        self.output_conv = nn.Conv1d(in_channels=h_u // 2**len(k_u), out_channels=1, kernel_size=7, padding="same")

    def forward(self, mel_spectrogram, **batch):
        """
        input: tensor of shape [B, n_mels, T']
        """
        x = self.input_conv(mel_spectrogram)
        for conv in self.t_convs:
            x = conv(x)

        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.output_conv(x)
        x = F.tanh(x)

        return {"generated_audio": x}
