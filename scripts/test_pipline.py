# test_pipeline.py
import sys, os
sys.path.append(os.path.abspath("../src"))

import torch
from configs.base import Config
from data.dataloader import build_train_test_dataset
from models.networks import TriMemoCMT
from utils.configs import get_options

cfg = get_options("../src/configs/mae_base.py")
cfg.raw_root = os.path.abspath("../rawdata")
cfg.processed_root = os.path.abspath("../data")
cfg.num_workers = 0

# Force CPU
device = torch.device("cpu")
network = TriMemoCMT(cfg, device="cpu")
network.to(device)

train_dl, _ = build_train_test_dataset(cfg)

# Grab just one batch
batch = next(iter(train_dl))
print("Batch keys:", batch.keys())
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape} {v.dtype}")

# Forward pass only — no backward
with torch.no_grad():
    text = batch["text"].to(device)
    audio = batch["audio"].to(device)
    video = batch["video"].to(device)
    speaker_ids = batch["speaker_ids"].to(device)
    mask = (batch["mask"] == 0).to(device)

    out, _ = network(text, audio, video,
                     speaker_ids=speaker_ids,
                     padding_mask=mask)
    print(f"Output shape: {out.shape}")
    print("Pipeline OK!")