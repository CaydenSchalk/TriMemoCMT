import logging
import os
import sys
from pathlib import Path

lib_path = os.path.abspath("").replace("scripts", "src")
sys.path.append(lib_path)

import argparse
import datetime
import random

import numpy as np
import torch
from torch import optim

import trainer as Trainer
from configs.base import Config
from data.dataloader import build_train_test_dataset
from models import losses, networks, optims
from utils.configs import get_options
from utils.torch.callbacks import CheckpointsCallback

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(cfg: Config):
    logging.info("Initializing model...")
    try:
        network = getattr(networks, cfg.model_type)(cfg)
        network.to(device)
    except AttributeError:
        raise NotImplementedError(f"Model {cfg.model_type} is not implemented")

    logging.info("Initializing checkpoint directory and dataset...")

    checkpoint_root = Path(cfg.checkpoint_root).expanduser().resolve()
    run_dir = checkpoint_root / cfg.name / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = run_dir / "logs"
    weight_dir = run_dir / "weights"

    log_dir.mkdir(parents=True, exist_ok=True)
    weight_dir.mkdir(parents=True, exist_ok=True)

    cfg.checkpoint_dir = str(run_dir)
    cfg.log_dir = str(log_dir)
    cfg.weight_dir = str(weight_dir)

    cfg.save(cfg)

    try:
        criterion = getattr(losses, cfg.loss_type)(cfg)
        criterion.to(device)
    except AttributeError:
        raise NotImplementedError(f"Loss {cfg.loss_type} is not implemented")

    try:
        trainer = getattr(Trainer, cfg.trainer)(
            cfg=cfg,
            network=network,
            criterion=criterion,
            log_dir=cfg.checkpoint_dir,
        )
    except AttributeError:
        raise NotImplementedError(f"Trainer {cfg.trainer} is not implemented")

    train_ds, test_ds = build_train_test_dataset(cfg)
    logging.info("Initializing trainer...")
    logging.info("Start training...")

    optimizer = optims.get_optim(cfg, network)
    lr_scheduler = None
    if cfg.learning_rate_step_size is not None:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.learning_rate_step_size,
            gamma=cfg.learning_rate_gamma,
        )

    ckpt_callback = CheckpointsCallback(
        checkpoint_dir=str(weight_dir),
        save_freq=cfg.save_freq,
        max_to_keep=cfg.max_to_keep,
        save_best_val=cfg.save_best_val,
        save_all_states=cfg.save_all_states,
    )

    if cfg.resume:
        trainer.load_all_states(cfg.resume_path)

    trainer.compile(optimizer=optimizer, scheduler=lr_scheduler)
    trainer.fit(train_ds, cfg.num_epochs, test_ds, callbacks=[ckpt_callback])


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, default="../src/configs/base.py")

    # runtime paths
    parser.add_argument("--raw_root", type=str, required=True)
    parser.add_argument("--processed_root", type=str, required=True)
    parser.add_argument("--checkpoint_root", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg: Config = get_options(args.config)

    if cfg.resume and cfg.cfg_path is not None:
        resume = cfg.resume
        resume_path = cfg.resume_path
        cfg.load(cfg.cfg_path)
        cfg.resume = resume
        cfg.resume_path = resume_path

    cfg.raw_root = str(Path(args.raw_root).expanduser().resolve())
    cfg.processed_root = str(Path(args.processed_root).expanduser().resolve())
    cfg.checkpoint_root = str(Path(args.checkpoint_root).expanduser().resolve())

    main(cfg)