import logging
import os
from typing import Dict

import torch
from torch import Tensor
from configs.base import Config
from models.networks import TriMemoCMT
from utils.torch.trainer import TorchTrainer


class Trainer(TorchTrainer):
    def __init__(
        self,
        cfg: Config,
        network: TriMemoCMT,
        criterion: torch.nn.CrossEntropyLoss = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.network = network
        # ignore_index=-100 so padded utterances don't contribute to loss
        self.criterion = criterion or torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

    def _unpack_batch(self, batch):
        """Move batch dict to device and return components."""
        text = batch["text"].to(self.device)           # (B, T_conv, seq_len)
        audio = batch["audio"].to(self.device)          # (B, T_conv, audio_len)
        video = batch["video"].to(self.device)          # (B, T_conv, F, C, H, W)
        labels = batch["labels"].to(self.device)        # (B, T_conv)
        mask = batch["mask"].to(self.device)            # (B, T_conv) — 1=real, 0=pad
        speaker_ids = batch.get("speaker_ids")
        if speaker_ids is not None:
            speaker_ids = speaker_ids.to(self.device)   # (B, T_conv)

        # TransformerEncoder expects True=ignore, but our mask is 1=real
        padding_mask = (mask == 0)  # True where padded

        return text, audio, video, labels, speaker_ids, padding_mask, mask

    def _compute_loss_and_acc(self, logits, labels, mask):
        """
        logits: (B, T_conv, num_classes)
        labels: (B, T_conv) — padded positions are -100
        mask:   (B, T_conv) — 1=real, 0=pad
        """
        B, T_conv, C = logits.shape
        # Flatten for cross-entropy
        logits_flat = logits.view(B * T_conv, C)
        labels_flat = labels.view(B * T_conv)

        loss = self.criterion(logits_flat, labels_flat)

        # Accuracy over real utterances only
        preds = logits_flat.argmax(dim=1)
        real_mask = (labels_flat != -100)
        correct = ((preds == labels_flat) & real_mask).sum()
        total = real_mask.sum()
        accuracy = correct.float() / total.float() if total > 0 else torch.tensor(0.0)

        return loss, accuracy

    # def train_step(self, batch):
    #     self.network.train()
    #     self.optimizer.zero_grad()
    #
    #     text, audio, video, labels, speaker_ids, padding_mask, mask = self._unpack_batch(batch)
    #
    #     output, _ = self.network(text, audio, video,
    #                               speaker_ids=speaker_ids,
    #                               padding_mask=padding_mask)
    #
    #     loss, accuracy = self._compute_loss_and_acc(output, labels, mask)
    #
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     return {
    #         "loss": loss.detach().cpu().item(),
    #         "acc": accuracy.detach().cpu().item(),
    #     }

    def train_step(self, batch):
        self.network.train()

        text         = batch["text"].to(self.device)          # (B, T_conv, seq_len)
        audio        = batch["audio"].to(self.device)         # (B, T_conv, audio_len)
        video        = batch["video"].to(self.device)         # (B, T_conv, F, C, H, W)
        labels       = batch["labels"].to(self.device)        # (B, T_conv)
        speaker_ids  = batch["speaker_ids"].to(self.device)   # (B, T_conv)
        mask         = batch["mask"].to(self.device)          # (B, T_conv)

        B, T_conv = text.shape[:2]
        hidden_dim = self.cfg.fusion_dim

        # initialise hidden states fresh per conversation
        global_state   = torch.zeros(B, hidden_dim).to(self.device)
        speaker_states = torch.zeros(B, 2, hidden_dim).to(self.device)

        total_loss = 0.0
        total_correct = 0
        total_real = 0

        for t in range(T_conv):
            label = labels[:, t]
            if (label == -100).all():  # fully padded timestep, stop early
                break

            self.optimizer.zero_grad()

            out, global_state, speaker_states = self.network(
                text[:, t, :],
                audio[:, t, :],
                video[:, t, :],
                speaker_ids[:, t],
                global_state,
                speaker_states,
            )

            # detach so gradients don't flow into previous utterances
            global_state   = global_state.detach()
            speaker_states = speaker_states.detach()

            # only compute loss on real utterances
            real = (label != -100)
            if real.any():
                loss = self.criterion(out[real], label[real])
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_correct += (out[real].argmax(dim=1) == label[real]).sum().item()
                total_real += real.sum().item()

        return {
            "loss": total_loss / max(total_real, 1),
            "acc": total_correct / max(total_real, 1),
        }

    # def test_step(self, batch):
    #     self.network.eval()
    #
    #     text, audio, video, labels, speaker_ids, padding_mask, mask = self._unpack_batch(batch)
    #
    #     with torch.no_grad():
    #         output, _ = self.network(text, audio, video,
    #                                   speaker_ids=speaker_ids,
    #                                   padding_mask=padding_mask)
    #
    #         loss, accuracy = self._compute_loss_and_acc(output, labels, mask)
    #
    #     return {
    #         "loss": loss.detach().cpu().item(),
    #         "acc": accuracy.detach().cpu().item(),
    #     }

    def test_step(self, batch):
        self.network.eval()

        text        = batch["text"].to(self.device)
        audio       = batch["audio"].to(self.device)
        video       = batch["video"].to(self.device)
        labels      = batch["labels"].to(self.device)
        speaker_ids = batch["speaker_ids"].to(self.device)

        B, T_conv = text.shape[:2]
        hidden_dim = self.cfg.fusion_dim

        global_state   = torch.zeros(B, hidden_dim).to(self.device)
        speaker_states = torch.zeros(B, 2, hidden_dim).to(self.device)

        total_loss    = 0.0
        total_correct = 0
        total_real    = 0

        with torch.no_grad():
            for t in range(T_conv):
                label = labels[:, t]
                if (label == -100).all():
                    break

                out, global_state, speaker_states = self.network(
                    text[:, t, :],
                    audio[:, t, :],
                    video[:, t, :],
                    speaker_ids[:, t],
                    global_state,
                    speaker_states,
                )

                real = (label != -100)
                if real.any():
                    loss = self.criterion(out[real], label[real])
                    total_loss    += loss.item()
                    total_correct += (out[real].argmax(dim=1) == label[real]).sum().item()
                    total_real    += real.sum().item()

        return {
            "loss": total_loss / max(total_real, 1),
            "acc":  total_correct / max(total_real, 1),
        }