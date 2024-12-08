import hydra
import torch
import torchaudio
from tqdm import tqdm
from hydra.utils import instantiate
import os

from src.utils.io_utils import ROOT_PATH

from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(config):
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
    processor = bundle.get_text_processor()
    tacotron2 = bundle.get_tacotron2().to(device)

    generator = instantiate(config.model).to(device)

    if config.from_pretrained is not None:
        checkpoint = torch.load(ROOT_PATH / config.from_pretrained, device)

        if checkpoint.get("generator_state_dict") is not None:
            generator.load_state_dict(checkpoint["generator_state_dict"])
        else:
            generator.load_state_dict(checkpoint)


    generator.eval()
    output_path = ROOT_PATH / config.output_path

    if config.text is not None:
        with torch.inference_mode():
            processed, lengths = processor(config.text)
            processed = processed.to(device)
            lengths = lengths.to(device)
            mel_spectrogram, _, _ = tacotron2.infer(processed, lengths)
        with torch.no_grad():
            generated_audio = generator(mel_spectrogram.cpu().detach())["generated_audio"]
            torchaudio.save(output_path / f"{config.file_name}.wav", generated_audio.squeeze(1).cpu().detach(), sample_rate=22050)
    else:
        input_path = ROOT_PATH / config.input_path / 'transcriptions'
        for gt_text_path in tqdm(os.listdir(input_path)):
            with open(input_path / gt_text_path, "r") as f:
                text = f.read()
            with torch.inference_mode():
                processed, lengths = processor(text)
                processed = processed.to(device)
                lengths = lengths.to(device)
                mel_spectrogram, _, _ = tacotron2.infer(processed, lengths)
            with torch.no_grad():
                generated_audio = generator(mel_spectrogram.cpu().detach())["generated_audio"]
                torchaudio.save(output_path / f"{gt_text_path}.wav", generated_audio.squeeze(1).cpu().detach(), sample_rate=22050)


    '''
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
        torchaudio.save(output_path / gt_audio_path, generated_audio.squeeze(1).detach().cpu(), sample_rate=22050)'''

if __name__ == "__main__":
    main()
