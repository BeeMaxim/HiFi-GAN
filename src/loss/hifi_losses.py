import torch
from torch import nn
import torch.nn.functional as F


def feature_matching_loss(audio_maps, generated_audio_maps):
    loss = 0
    for maps_a, maps_g in zip(audio_maps, generated_audio_maps):
        for map_a, map_g in zip(maps_a, maps_g):
            loss += torch.mean(torch.abs(map_a - map_g))
    return loss


def generator_loss(generated_audio_preds):
    loss = 0
    for pred_g in generated_audio_preds:
        loss += torch.mean((1 - pred_g)**2)
    return loss


class DiscriminatorLoss(nn.Module):
    """
    Loss for discriminator
    """

    def __init__(self):
        super().__init__()

    def forward(self, audio_preds, generated_audio_preds, **batch):
        loss = 0
        for pred_a, pred_g in zip(audio_preds, generated_audio_preds):
            loss += torch.mean((1 - pred_a)**2) + torch.mean(pred_g**2)
        return {"discriminator_loss": loss}


class GeneratorLoss(nn.Module):
    """
    Weighted sum of losses for generator
    """

    def __init__(self):
        super().__init__()

    def forward(self, generated_audio_preds, mel_spectrogram, melspec_after, audio_maps, generated_audio_maps, **batch):
        f_loss = feature_matching_loss(audio_maps, generated_audio_maps) 
        g_loss = generator_loss(generated_audio_preds)
        mel_loss = F.l1_loss(mel_spectrogram, melspec_after)
        loss = 2 * f_loss + g_loss + 45 * mel_loss
        return {"feature_loss": f_loss, "g_loss": g_loss, "mel_loss": mel_loss, "generator_loss": loss}
