import json
import logging
import os
import shutil
import wget
from pathlib import Path

import torch
import torchaudio
import numpy as np
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH
from tqdm import tqdm

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dataset": "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2", 
}


class LJspeechDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "ljspeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_dataset(self):
        arch_path = self._data_dir / "LJSpeech-1.1.tar.bz2"
        print(f"Loading LJSpeech")
        wget.download(URL_LINKS["dataset"], str(arch_path))
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LJSpeech-1.1").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LJSpeech-1.1"))

        files = [file_name for file_name in (self._data_dir / "wavs").iterdir()]
        train_length = int(0.85 * len(files)) # hand split, test ~ 15% 
        (self._data_dir / "train").mkdir(exist_ok=True, parents=True)
        (self._data_dir / "test").mkdir(exist_ok=True, parents=True)
        for i, fpath in enumerate((self._data_dir / "wavs").iterdir()):
            if i < train_length:
                shutil.move(str(fpath), str(self._data_dir / "train" / fpath.name))
            else:
                shutil.move(str(fpath), str(self._data_dir / "test" / fpath.name))
        shutil.rmtree(str(self._data_dir / "wavs"))


    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_dataset()

        wav_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".wav") for f in filenames]):
                wav_dirs.add(dirpath)
        for wav_dir in tqdm(
                list(wav_dirs), desc=f"Preparing ljspeech folders: {part}"
        ):
            wav_dir = Path(wav_dir)
            trans_path = list(self._data_dir.glob("*.csv"))[0]
            with trans_path.open() as f:
                for line in f:
                    w_id = line.split('|')[0]
                    w_text = " ".join(line.split('|')[1:]).strip()
                    wav_path = wav_dir / f"{w_id}.wav"
                    if not wav_path.exists(): # elem in another part
                        continue
                    t_info = torchaudio.info(str(wav_path))
                    length = t_info.num_frames / t_info.sample_rate
                    if w_text.isascii():
                        index.append(
                            {
                                "path": str(wav_path.absolute().resolve()),
                                "text": w_text.lower(),
                                "audio_len": length,
                            }
                        )
        return index
    
    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        text = data_dict["text"]
        audio_len = data_dict["audio_len"]

        audio = self.load_audio(audio_path)
        if audio.shape[1] > 8192:
            audio = audio[:, :8192]

        instance_data = {"audio_path": audio_path, "text": text,
                         "audio_len": audio_len, "audio": audio}
        return instance_data
    