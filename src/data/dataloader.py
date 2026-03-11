import os
import pickle
import re
from typing import Tuple, Union
import logging
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    VideoMAEImageProcessor,
    AutoModel,
    AutoConfig
)
import torchaudio

from models.networks import TriMemoCMT
from configs.base import Config
from torchvggish.vggish_input import waveform_to_examples
from tqdm.auto import tqdm
import pickle
import cv2
import librosa


class BaseDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        data_mode: str = "train.pkl",
        encoder_model: Union[TriMemoCMT, None] = None,
    ):
        """Dataset for IEMOCAP

        Args:
            path (str, optional): Path to data.pkl. Defaults to "path/to/data.pkl".
            encoder_model (_4M_SER, optional): if want to pre-encoder dataset
        """
        super(BaseDataset, self).__init__()

        with open(os.path.join(cfg.data_root, data_mode), "rb") as train_file:
            self.data_list = pickle.load(train_file)

        if cfg.text_encoder_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            raise NotImplementedError(
                "Tokenizer {} is not implemented".format(cfg.text_encoder_type)
            )

        self.audio_max_length = cfg.audio_max_length
        self.text_max_length = cfg.text_max_length
        self.video_max_length = cfg.video_max_length

        self.video_processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Base")

        if cfg.batch_size == 1:
            self.audio_max_length = None
            self.text_max_length = None
            self.video_max_length = None

        self.audio_encoder_type = cfg.audio_encoder_type

        self.encode_data = False
        self.list_encode_audio_data = []
        self.list_encode_text_data = []
        self.list_encode_video_data = []
        if encoder_model is not None:
            self._encode_data(encoder_model)
            self.encode_data = True

    def _encode_data(self, encoder):
        logging.info("Encoding data for training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.train()
        encoder.to(device)
        for index in tqdm(range(len(self.data_list))):
            video_path, audio_path, text, _ = self.data_list[index]

            samples = self.__paudio__(audio_path)
            audio_embedding = (
                encoder.encode_audio(samples.unsqueeze(0).to(device))
                .squeeze(0)
                .detach()
                .cpu()
            )
            self.list_encode_audio_data.append(audio_embedding)

            frames = self.__pvideo__(video_path)
            video_embedding = (
                encoder.encode_video(frames.unsqueeze(0).to(device))
                .squeeze(0)
                .detach()
                .cpu()
            )
            self.list_encode_video_data.append(video_embedding)

            input_ids = self.__ptext__(text)
            text_embedding = (
                encoder.encode_text(input_ids.unsqueeze(0).to(device))
                .squeeze(0)
                .detach()
                .cpu()
            )
            self.list_encode_text_data.append(text_embedding)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        video_path, audio_path, text, label = self.data_list[index]
        input_audio = (
            self.list_encode_audio_data[index]
            if self.encode_data
            else self.__paudio__(audio_path)
        )
        input_text = (
            self.list_encode_text_data[index]
            if self.encode_data
            else self.__ptext__(text)
        )
        label = self.__plabel__(label)

        input_video = (
            self.list_encode_video_data[index]
            if self.encode_data
            else self.__pvideo__(video_path)
        )

        return input_video, input_text, input_audio, label

    def __pvideo__(self, file_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {file_path}")

        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"Empty video: {file_path}")

        T = self.video_max_length if self.video_max_length is not None else 16
        idx = np.linspace(0, len(frames) - 1, T).astype(int)
        frames = [frames[i] for i in idx] # list of [H, W, C] numpy arrays

        inputs = self.video_processor(frames, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)  # remove batch dim, dataloader will re-add it

    def __paudio__(self, file_path: str) -> torch.Tensor:
        samples, sr = sf.read(file_path, dtype="float32")

        # collapse to mono channel, take the average across the channel axis
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # since time is now the dimension, check if the shape is 0, we can see if it's empty
        if samples.shape[0] == 0:
            raise ValueError(f"Empty audio file: {file_path}")

        x = torch.from_numpy(samples)

        # resample to 16kHz, expected by hubert
        if sr != 16000:
            x = torchaudio.functional.resample(x, sr, 16000)

        # enforce minimum length of 400, hubert needs this for the front-end conv layers
        min_len = 400
        if x.numel() < min_len:
            x = torch.nn.functional.pad(x, (0, min_len - x.numel()))

        # ensure it stays within max length
        if self.audio_max_length is not None:
            if x.numel() < self.audio_max_length:
                x = torch.nn.functional.pad(x, (0, self.audio_max_length - x.numel()))
            else:
                x = x[: self.audio_max_length]

        # just return as a float
        return x.float()

    def _text_preprocessing(self, text):
        """
        - Remove entity mentions (eg. '@united')
        - Correct errors (eg. '&amp;' to '&')
        @param    text (str): a string to be processed.
        @return   text (Str): the processed string.
        """
        # Remove '@name'
        text = re.sub("[\(\[].*?[\)\]]", "", text)

        # Replace '&amp;' with '&'
        text = re.sub(" +", " ", text).strip()

        # Normalize and clean up text; order matters!
        try:
            text = " ".join(text.split())  # clean up whitespaces
        except:
            text = "NULL"

        # Convert empty string to NULL
        if not text.strip():
            text = "NULL"

        return text

    def __ptext__(self, text: str) -> torch.Tensor:
        text = self._text_preprocessing(text)
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        if self.text_max_length is not None and len(input_ids) < self.text_max_length:
            input_ids = np.pad(
                input_ids,
                (0, self.text_max_length - len(input_ids)),
                "constant",
                constant_values=self.tokenizer.pad_token_id,
            )
        elif self.text_max_length is not None:
            input_ids = input_ids[: self.text_max_length]
        return torch.from_numpy(np.asarray(input_ids))

    def __plabel__(self, label: int) -> torch.Tensor:
        return torch.tensor(label)

    def __len__(self):
        return len(self.data_list)


def build_train_test_dataset(cfg: Config, encoder_model: Union[TriMemoCMT, None] = None):
    DATASET_MAP = {
        "IEMOCAP": BaseDataset,
        "ESD": BaseDataset,
        "MELD" : BaseDataset,
    }

    dataset = DATASET_MAP.get(cfg.data_name, None)
    if dataset is None:
        raise NotImplementedError(
            "Dataset {} is not implemented, list of available datasets: {}".format(
                cfg.data_name, DATASET_MAP.keys()
            )
        )
    if cfg.data_name in ["IEMOCAP_MSER", "MELD_MSER"]:
        return dataset(cfg)

    train_data = dataset(
        cfg,
        data_mode="train.pkl",
        encoder_model=encoder_model,
    )

    if encoder_model is not None:
        encoder_model.eval()
    test_set = cfg.data_valid if cfg.data_valid is not None else "test.pkl"
    test_data = dataset(
        cfg,
        data_mode=test_set,
        encoder_model=encoder_model,
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return (train_dataloader, test_dataloader)
