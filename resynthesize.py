import warnings

import hydra
import torch
import torchaudio
from tqdm import tqdm
from hydra.utils import instantiate
import os
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.io_utils import ROOT_PATH

from src.model.discriminators import Discriminator
from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig
from src.datasets.ljspeech_dataset import LJspeechDataset


@hydra.main(version_base=None, config_path="src/configs", config_name="resynthesize")
def main(config):
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    generator = instantiate(config.model).to(device)
    if config.from_pretrained is not None:
        checkpoint = torch.load(ROOT_PATH / config.from_pretrained, device)

        if checkpoint.get("generator_state_dict") is not None:
            generator.load_state_dict(checkpoint["generator_state_dict"])
        else:
            generator.load_state_dict(checkpoint)
    melspec = MelSpectrogram(MelSpectrogramConfig).to(device)


    input_path = ROOT_PATH / config.input_path
    output_path = ROOT_PATH / config.output_path

    for gt_audio_path in tqdm(os.listdir(config.input_path)):
        audio, sr = torchaudio.load(input_path / gt_audio_path)
        audio = audio[0:1, :].to(device)
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        mel_spectrogram = melspec(audio)
        generated_audio = generator(mel_spectrogram)["generated_audio"]
        torchaudio.save(output_path / gt_audio_path, generated_audio.squeeze(1).detach().cpu(), sample_rate=22050)

if __name__ == "__main__":
    main()
