import torch
import torch.nn.functional as F
import torch.nn as nn

LRELU_SLOPE = 0.1


class MPD(nn.Module):
    def __init__(self, p):
        super().__init__()

        self.p = p

        self.convs = nn.ModuleList()
        channels = 1
        for l in range(1, 5):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=2**(5 + l), kernel_size=(5, 1), stride=(3, 1)),
                nn.LeakyReLU(LRELU_SLOPE)
            ))
            channels = 2**(5 + l)
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1024, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(3, 1), padding="same")
        )

    def forward(self, x):
        maps = []
        B, C, T = x.shape

        pad = self.p - T % self.p
        x = F.pad(x, (0, pad))
        x = x.view(B, C, (T + pad) // self.p, self.p)

        for conv in self.convs:
            x = conv(x)
            maps.append(x)
      
        x = self.output_conv(x)
        maps.append(x)

        x = torch.flatten(x, 1, -1)
        return x, maps
    

class MSD(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.ModuleList()

        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15, padding="same")
        )

        channels = 16
        for l in range(1, 5):
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels=channels, out_channels=min(1024, channels * 4), kernel_size=41, stride=4, groups=4**l, padding=20),
                nn.LeakyReLU(LRELU_SLOPE)
            ))
            channels = min(1024, channels * 4)

        self.output_conv = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=5, padding="same"),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(in_channels=1024, out_channels=1, kernel_size=3, padding="same")
        )

    def forward(self, x):
        maps = []
        
        x = self.input_conv(x)
        maps.append(x)

        for conv in self.convs:
            x = conv(x)
            maps.append(x)
        
        x = self.output_conv(x)
        maps.append(x)

        x = torch.flatten(x, 1, -1)
        return x, maps
    

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.mpd = nn.ModuleList([
            MPD(2),
            MPD(3),
            MPD(5),
            MPD(7),
            MPD(11)
        ])
        self.msd = nn.ModuleList([
            MSD(),
            MSD(),
            MSD()
        ])
        self.poolings = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, audio, generated_audio, **batch):
        audio_preds = []
        audio_maps = []
        generated_audio_preds = []
        generated_audio_maps = []

        for mpd in self.mpd:
            pred_a, maps_a = mpd(audio)
            pred_g, maps_g = mpd(generated_audio)

            audio_preds.append(pred_a)
            audio_maps.append(maps_a)
            generated_audio_preds.append(pred_g)
            generated_audio_maps.append(maps_g)

        for i in range(3):
            if i > 0:
                audio = F.avg_pool1d(audio, kernel_size=4, stride=2, padding=2)
                generated_audio = F.avg_pool1d(generated_audio, kernel_size=4, stride=2, padding=2)
            pred_a, maps_a = self.msd[i](audio)
            pred_g, maps_g = self.msd[i](generated_audio)

            audio_preds.append(pred_a)
            audio_maps.append(maps_a)
            generated_audio_preds.append(pred_g)
            generated_audio_maps.append(maps_g)

        return {"audio_preds": audio_preds, "generated_audio_preds": generated_audio_preds, 
                "audio_maps": audio_maps, "generated_audio_maps": generated_audio_maps}
