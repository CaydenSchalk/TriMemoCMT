import os
import pickle
import re
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import BertTokenizer, VideoMAEImageProcessor

from models.networks import TriMemoCMT
from configs.base import Config


class BaseDataset(Dataset):
    def __init__(
        self,
        cfg: Config,
        data_mode: str = "train.pkl",
        encoder_model: Union[TriMemoCMT, None] = None,
    ):
        super(BaseDataset, self).__init__()
        self.cfg = cfg

        # dataset root now comes from runtime processed_root + logical dataset name
        self.dataset_root = Path(cfg.processed_root) / cfg.data_name

        # raw root can either already point at the specific dataset root,
        # or at a parent folder containing dataset subfolders
        raw_root_path = Path(cfg.raw_root)
        if raw_root_path.name == cfg.data_name:
            self.raw_dataset_root = raw_root_path
        else:
            self.raw_dataset_root = raw_root_path / cfg.data_name

        split_path = self.dataset_root / data_mode
        with open(split_path, "rb") as train_file:
            self.data_list = pickle.load(train_file)

        if cfg.text_encoder_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        else:
            raise NotImplementedError(
                f"Tokenizer {cfg.text_encoder_type} is not implemented"
            )

        self.audio_max_length = cfg.audio_max_length
        self.text_max_length = cfg.text_max_length
        self.video_max_length = cfg.video_max_length
        self.video_processor = VideoMAEImageProcessor.from_pretrained(
            "OpenGVLab/VideoMAEv2-Base"
        )

        # if cfg.batch_size == 1:
        #     self.audio_max_length = None
        #     self.text_max_length = None
        #     self.video_max_length = None

        self.audio_encoder_type = cfg.audio_encoder_type

        # Disk cache setup
        split_name = Path(data_mode).stem  # "train", "val", "test"
        self._cache_dir = self.dataset_root / "cache" / split_name
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Config fingerprint - cache auto-invalidates when these change
        config_str = (
            f"audio_max={self.audio_max_length}|"
            f"text_max={self.text_max_length}|"
            f"video_max={self.video_max_length}|"
            f"audio_enc={self.audio_encoder_type}|"
            f"text_enc={cfg.text_encoder_type}|"
            f"batch_size={cfg.batch_size}"
        )
        self._config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Encoder embedding cache
        self.encode_data = False
        self.list_encode_audio_data = []
        self.list_encode_text_data = []
        self.list_encode_video_data = []
        if encoder_model is not None:
            self._encode_data(encoder_model)
            self.encode_data = True

    # Cache helpers

    def _sample_cache_key(self, index: int) -> str:
        """Deterministic cache key for a given sample index."""
        video_path, audio_path, text, label = self._resolve_sample(
            self.data_list[index]
        )
        raw = f"{video_path}|{audio_path}|{text}|{label}|{self._config_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_path(self, index: int) -> Path:
        key = self._sample_cache_key(index)
        return self._cache_dir / f"{key}.pt"

    def precache(self, num_workers: int = 0):
        """Warm the disk cache for every sample.

        Call once before training:
            dataset.precache()

        After this completes, every __getitem__ is just torch.load().
        """
        uncached = [
            i for i in range(len(self.data_list)) if not self._cache_path(i).exists()
        ]
        if not uncached:
            logging.info(
                f"Cache already warm ({len(self.data_list)} samples) at {self._cache_dir}"
            )
            return

        logging.info(
            f"Pre-caching {len(uncached)} / {len(self.data_list)} samples "
            f"to {self._cache_dir}"
        )
        for i in tqdm(uncached, desc="Caching"):
            self._preprocess_and_cache(i)
        logging.info("Pre-cache complete.")

    def _preprocess_and_cache(self, index: int) -> dict:
        """Run full preprocessing for one sample and persist to disk."""
        video_path, audio_path, text, label = self._resolve_sample(
            self.data_list[index]
        )

        video_tensor = self.__pvideo__(video_path)
        audio_tensor = self.__paudio__(audio_path)
        text_tensor = self.__ptext__(text)
        label_tensor = self.__plabel__(label)

        payload = {
            "video": video_tensor,
            "audio": audio_tensor,
            "text": text_tensor,
            "label": label_tensor,
        }

        cache_file = self._cache_path(index)
        tmp_file = cache_file.with_suffix(".tmp")
        torch.save(payload, tmp_file)
        tmp_file.rename(cache_file)  # atomic on same filesystem

        return payload

    # Sample resolution

    def _resolve_sample(self, sample):
        if isinstance(sample, dict):
            text = sample["text"]
            label = sample["label"]
            video_rel = sample.get("video_relpath")
            audio_rel = sample.get("audio_relpath")
            video_path = None
            audio_path = None
            if video_rel is not None:
                video_path = str(self.dataset_root / video_rel)
            if audio_rel is not None:
                audio_path = str(self.raw_dataset_root / audio_rel)
            return video_path, audio_path, text, label

        if isinstance(sample, (tuple, list)):
            if len(sample) != 4:
                raise ValueError(f"Unexpected sample format: {sample}")
            video_value, audio_value, text, label = sample
            if os.path.isabs(str(video_value)):
                video_path = str(video_value)
            else:
                video_path = str(self.dataset_root / str(video_value))
            if os.path.isabs(str(audio_value)):
                audio_path = str(audio_value)
            else:
                audio_path = str(self.raw_dataset_root / str(audio_value))
            return video_path, audio_path, text, label

        raise TypeError(f"Unsupported sample type: {type(sample)}")

    # Encoder embedding pre-computation

    def _encode_data(self, encoder):
        logging.info("Encoding data for training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder.train()
        encoder.to(device)

        for index in tqdm(range(len(self.data_list))):
            video_path, audio_path, text, _ = self._resolve_sample(
                self.data_list[index]
            )

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

    def __getitem__(self, index):
        conv = self.data_list[index]
        utterances = conv["utterances"]
        speaker_map = {"M": 0, "F": 1}
        dialog_id = conv.get("dialog_id", f"conv_{index}")

        videos, audios, texts, labels, speakers = [], [], [], [], []
        cache_hits = 0
        cache_misses = 0

        # Check how many need caching before starting
        needs_caching = 0
        for utt in utterances:
            raw = f"{utt['audio_relpath']}|{utt['video_relpath']}|{self._config_hash}"
            key = hashlib.sha256(raw.encode()).hexdigest()[:16]
            if not (self._cache_dir / f"{key}.pt").exists():
                needs_caching += 1

        if needs_caching > 0:
            print(f"[{dialog_id}] Building cache for {needs_caching}/{len(utterances)} utterances...")
            utt_iter = tqdm(utterances, desc=f"Caching {dialog_id}", leave=False)
        else:
            utt_iter = utterances

        for utt in utt_iter:
            cache_key = hashlib.sha256(
                f"{utt['audio_relpath']}|{utt['video_relpath']}|{self._config_hash}".encode()
            ).hexdigest()[:16]
            cache_file = self._cache_dir / f"{cache_key}.pt"

            if cache_file.exists():
                payload = torch.load(cache_file, weights_only=True)
                cache_hits += 1
            else:
                video_path = str(self.dataset_root / utt["video_relpath"])
                audio_path = str(self.raw_dataset_root / utt["audio_relpath"])
                payload = {
                    "video": self.__pvideo__(video_path),
                    "audio": self.__paudio__(audio_path),
                    "text": self.__ptext__(utt["text"]),
                }
                tmp = cache_file.with_suffix(".tmp")
                torch.save(payload, tmp)
                tmp.rename(cache_file)
                cache_misses += 1

            videos.append(payload["video"])
            audios.append(payload["audio"])
            texts.append(payload["text"])
            labels.append(self.__plabel__(utt["label"]))
            speakers.append(speaker_map[utt["speaker"]])

        print(f"[{dialog_id}] {len(utterances)} utts | cache: {cache_hits} hits, {cache_misses} misses")

        return {
            "video": torch.stack(videos),
            "audio": torch.stack(audios),
            "text": torch.stack(texts),
            "labels": torch.stack(labels),
            "speaker_ids": torch.tensor(speakers, dtype=torch.long),
            "lengths": len(utterances),
        }

    # def __getitem__(self, index):
    #     conv = self.data_list[index]
    #     utterances = conv["utterances"]
    #
    #     speaker_map = {"M": 0, "F": 1}
    #
    #     videos, audios, texts, labels, speakers = [], [], [], [], []
    #     for utt in utterances:
    #         video_path = str(self.dataset_root / utt["video_relpath"])
    #         audio_path = str(self.raw_dataset_root / utt["audio_relpath"])
    #
    #         videos.append(self.__pvideo__(video_path))
    #         audios.append(self.__paudio__(audio_path))
    #         texts.append(self.__ptext__(utt["text"]))
    #         labels.append(self.__plabel__(utt["label"]))
    #         speakers.append(speaker_map[utt["speaker"]])
    #
    #     return {
    #         "video": torch.stack(videos),
    #         "audio": torch.stack(audios),
    #         "text": torch.stack(texts),
    #         "labels": torch.stack(labels),
    #         "speaker_ids": torch.tensor(speakers, dtype=torch.long),
    #         "lengths": len(utterances),
    #     }

    # def __getitem__(
    #     self, index: int
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #
    #     # Fast path: pre-computed encoder embeddings (stage-2)
    #     if self.encode_data:
    #         _, _, _, label = self._resolve_sample(self.data_list[index])
    #         return (
    #             self.list_encode_video_data[index],
    #             self.list_encode_text_data[index],
    #             self.list_encode_audio_data[index],
    #             self.__plabel__(label),
    #         )
    #
    #     # Normal path: disk-cached preprocessed tensors
    #     cache_file = self._cache_path(index)
    #     if cache_file.exists():
    #         payload = torch.load(cache_file)
    #         return (
    #             payload["video"],
    #             payload["text"],
    #             payload["audio"],
    #             payload["label"],
    #         )
    #
    #     # Cache miss — preprocess, persist, return
    #     payload = self._preprocess_and_cache(index)
    #     return (
    #         payload["video"],
    #         payload["text"],
    #         payload["audio"],
    #         payload["label"],
    #     )

    # Preprocessing helpers

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
        frames = [frames[i] for i in idx]

        inputs = self.video_processor(frames, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    def __paudio__(self, file_path: str) -> torch.Tensor:
        samples, sr = sf.read(file_path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if samples.shape[0] == 0:
            raise ValueError(f"Empty audio file: {file_path}")

        x = torch.from_numpy(samples)
        if sr != 16000:
            x = torchaudio.functional.resample(x, sr, 16000)

        min_len = 400
        if x.numel() < min_len:
            x = torch.nn.functional.pad(x, (0, min_len - x.numel()))

        if self.audio_max_length is not None:
            if x.numel() < self.audio_max_length:
                x = torch.nn.functional.pad(
                    x, (0, self.audio_max_length - x.numel())
                )
            else:
                x = x[: self.audio_max_length]

        return x.float()

    def _text_preprocessing(self, text):
        text = re.sub(r"[\(\[].*?[\)\]]", "", text)
        text = re.sub(" +", " ", text).strip()
        try:
            text = " ".join(text.split())
        except:
            text = "NULL"
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

def conversation_collate(batch):
    max_conv_len = max(b["lengths"] for b in batch)

    padded = {"video": [], "audio": [], "text": [], "labels": [], "mask": [], "speaker_ids": []}

    for b in batch:
        T = b["lengths"]
        pad_len = max_conv_len - T

        mask = torch.cat([torch.ones(T), torch.zeros(pad_len)])
        padded["mask"].append(mask)

        for key in ["video", "audio", "text"]:
            tensor = b[key]
            if pad_len > 0:
                pad_shape = (pad_len, *tensor.shape[1:])
                tensor = torch.cat([tensor, torch.zeros(pad_shape)], dim=0)
            padded[key].append(tensor)

        # Labels: pad with -100
        lab = b["labels"]
        if pad_len > 0:
            lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=lab.dtype)])
        padded["labels"].append(lab)

        # Speaker IDs: pad with 0 (value doesn't matter, masked out)
        spk = b["speaker_ids"]
        if pad_len > 0:
            spk = torch.cat([spk, torch.zeros(pad_len, dtype=spk.dtype)])
        padded["speaker_ids"].append(spk)

    return {k: torch.stack(v) for k, v in padded.items()}

def build_train_test_dataset(
    cfg: Config,
    encoder_model: Union[TriMemoCMT, None] = None,
    precache: bool = False,
):
    DATASET_MAP = {
        "IEMOCAP": BaseDataset,
        "ESD": BaseDataset,
        "MELD": BaseDataset,
    }
    dataset = DATASET_MAP.get(cfg.data_name, None)
    if dataset is None:
        raise NotImplementedError(
            "Dataset {} is not implemented, list of available datasets: {}".format(
                cfg.data_name, DATASET_MAP.keys()
            )
        )

    train_data = dataset(
        cfg,
        data_mode=cfg.data_train,
        encoder_model=encoder_model,
    )

    if encoder_model is not None:
        encoder_model.eval()

    test_set = cfg.data_valid if cfg.data_valid is not None else cfg.data_test
    test_data = dataset(
        cfg,
        data_mode=test_set,
        encoder_model=encoder_model,
    )

    if precache:
        logging.info("Warming disk cache for train + val/test splits...")
        train_data.precache()
        test_data.precache()

    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=conversation_collate,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=conversation_collate,
    )
    return train_dataloader, test_dataloader