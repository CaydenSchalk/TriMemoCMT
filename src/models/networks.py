import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from configs.base import Config
from .modules import build_audio_encoder, build_text_encoder, build_video_encoder


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        attn_out, attn_weights = self.attention(
            query, key_value, key_value,
            average_attn_weights=False,
        )
        out = self.dropout(self.layer_norm(self.linear(attn_out)))
        return out, attn_weights


class MultimodalFusion(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, shared_dim, num_heads, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, shared_dim)
        self.audio_proj = nn.Linear(audio_dim, shared_dim)
        self.video_proj = nn.Linear(video_dim, shared_dim)

        self.text_cross = CrossAttentionBlock(shared_dim, num_heads, dropout)
        self.audio_cross = CrossAttentionBlock(shared_dim, num_heads, dropout)
        self.video_cross = CrossAttentionBlock(shared_dim, num_heads, dropout)

    def forward(self, text, audio, video):
        # projection layers for alignment
        text = self.text_proj(text)
        audio = self.audio_proj(audio)
        video = self.video_proj(video)

        # text attends to audio and video
        text_audio, _ = self.text_cross(text, audio)
        text_video, _ = self.text_cross(text, video)

        # audio attends to text and video
        audio_text, _ = self.audio_cross(audio, text)
        audio_video, _ = self.audio_cross(audio, video)

        # video attends to audio and text
        video_audio, _ = self.video_cross(video, audio)
        video_text, _ = self.video_cross(video, text)

        return text_audio, text_video, audio_text, audio_video, video_audio, video_text


class TriMemoCMT(nn.Module):
    def __init__(self, cfg: Config, device: str = "cpu"):
        super(TriMemoCMT, self).__init__()

        # Text encoder
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        for param in self.text_encoder.parameters():
            param.requires_grad = cfg.text_unfreeze

        # Audio encoder
        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)
        for param in self.audio_encoder.parameters():
            param.requires_grad = cfg.audio_unfreeze

        # Video encoder
        self.video_encoder = build_video_encoder(cfg)
        self.video_encoder.to(device)
        for param in self.video_encoder.parameters():
            param.requires_grad = False

        if cfg.video_unfreeze:
            for block in self.video_encoder.model.model.blocks[-cfg.video_unfreeze_amount:]:
                for p in block.parameters():
                    p.requires_grad = True

        # Video CLS token + self-attention
        self.video_cls_token = nn.Parameter(torch.randn(1, 1, cfg.video_encoder_dim))
        self.video_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.video_encoder_dim,
                nhead=cfg.num_attention_head,
                dropout=cfg.dropout,
                batch_first=True,
            ),
            num_layers=2,
        )

        # Fusion
        self.dropout = nn.Dropout(cfg.dropout)
        self.fusion_module = MultimodalFusion(
            cfg.text_encoder_dim,
            cfg.audio_encoder_dim,
            cfg.video_encoder_dim,
            cfg.fusion_dim,
            cfg.num_attention_head,
            cfg.dropout,
        )

        self.fusion_head_output_type = cfg.fusion_head_output_type
        self.linear_layer_output = cfg.linear_layer_output

        # Fusion output dim
        fusion_out_dim = (6 * cfg.fusion_dim
                          if cfg.fusion_head_output_type == "cls_concat"
                          else cfg.fusion_dim)

        # GRU cells
        # global GRU: tracks overall conversation state
        self.global_gru = nn.GRUCell(fusion_out_dim, cfg.fusion_dim)
        # speaker GRU: tracks each speaker's emotional trajectory
        self.speaker_gru = nn.GRUCell(fusion_out_dim + cfg.fusion_dim, cfg.fusion_dim)
        # emotion GRU: combines utterance + speaker context then feeds classifier
        self.emotion_gru = nn.GRUCell(fusion_out_dim + cfg.fusion_dim, cfg.fusion_dim)

        self.use_temporal = cfg.use_temporal

        # Classification head
        # when use_temporal=True:  input is emotion_gru output → cfg.fusion_dim
        # when use_temporal=False: input is fused utt_embed   → fusion_out_dim
        previous_dim = cfg.fusion_dim if cfg.use_temporal else fusion_out_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

    def _encode_single_video(self, video_chunk):
        v = self.video_encoder(video_chunk)
        cls = self.video_cls_token.expand(v.size(0), -1, -1)
        v = torch.cat([cls, v], dim=1)
        v = self.video_self_attn(v)
        return v

    def forward(
            self,
            utt_text,       # (B, seq_len)
            utt_audio,      # (B, audio_len)
            utt_video,      # (B, F, C, H, W)
            speaker_id,     # (B,)
            global_state,   # (B, fusion_dim)
            speaker_states, # (B, 2, fusion_dim)
    ):
        B = utt_text.size(0)

        # Encode
        with torch.no_grad():
            text_emb  = self.text_encoder(utt_text).last_hidden_state  # (B, seq_len, 768)
            audio_emb = self.audio_encoder(utt_audio)                  # (B, audio_frames, 768)

        video_emb = checkpoint(
            self._encode_single_video, utt_video, use_reentrant=False
        )  # (B, N+1, 768)

        # Cross-attention fusion
        text_audio, text_video, audio_text, audio_video, video_audio, video_text = \
            self.fusion_module(text_emb, audio_emb, video_emb)

        cls_tokens = torch.stack([
            text_audio[:, 0, :], text_video[:, 0, :],
            audio_text[:, 0, :], audio_video[:, 0, :],
            video_audio[:, 0, :], video_text[:, 0, :],
        ], dim=1)  # (B, 6, D)

        if self.fusion_head_output_type == "cls_concat":
            utt_embed = cls_tokens.reshape(B, -1)   # (B, 6*D)
        else:  # cls_mean
            utt_embed = cls_tokens.mean(dim=1)      # (B, D)

        # one RNN step
        if self.use_temporal:
            # Update global conversation state
            global_state = self.global_gru(utt_embed, global_state)            # (B, H)

            # Get this speaker's current state
            spk_state = speaker_states[torch.arange(B), speaker_id]            # (B, H)

            # Update speaker state with utterance + global context
            new_spk_state = self.speaker_gru(
                torch.cat([utt_embed, global_state], dim=-1), spk_state
            )                                                                   # (B, H)
            speaker_states[torch.arange(B), speaker_id] = new_spk_state

            # Emotion state: utterance + updated speaker context
            x = self.emotion_gru(
                torch.cat([utt_embed, new_spk_state], dim=-1),
                torch.zeros(B, self.emotion_gru.hidden_size, device=utt_embed.device),
            )                                                                   # (B, H)
        else:
            # No temporal, pass fused embedding directly
            x = utt_embed

        # Classification
        x = self.dropout(x)
        for i in range(len(self.linear_layer_output)):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifer(x)  # (B, num_classes)

        return out, global_state, speaker_states

    def encode_audio(self, audio: torch.Tensor):
        return self.audio_encoder(audio)

    def encode_text(self, input_ids: torch.Tensor):
        return self.text_encoder(input_ids).last_hidden_state

    def encode_video(self, input_ids: torch.Tensor):
        return self.video_encoder(input_ids)


class TextOnly(nn.Module):
    def __init__(self, cfg: Config, device: str = "cpu"):
        super(TextOnly, self).__init__()
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        for param in self.text_encoder.parameters():
            param.requires_grad = cfg.text_unfreeze

        self.dropout = nn.Dropout(cfg.dropout)
        self.linear_layer_output = cfg.linear_layer_output

        previous_dim = cfg.text_encoder_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)
        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(self, input_text, input_audio, output_attentions=False):
        text_embeddings = self.text_encoder(input_text).last_hidden_state
        fusion_norm = self.dropout(text_embeddings)

        if self.fusion_head_output_type == "cls":
            x = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            x = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            x = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            x = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        x = self.dropout(x)
        for i in range(len(self.linear_layer_output)):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifer(x)
        return out, x


class AudioOnly(nn.Module):
    def __init__(self, cfg: Config, device: str = "cpu"):
        super(AudioOnly, self).__init__()
        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)
        for param in self.audio_encoder.parameters():
            param.requires_grad = cfg.audio_unfreeze

        self.dropout = nn.Dropout(cfg.dropout)
        self.linear_layer_output = cfg.linear_layer_output

        previous_dim = cfg.audio_encoder_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)
        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(self, input_text, input_audio, output_attentions=False):
        if len(input_audio.size()) != 2:
            batch_size, num_samples = input_audio.size(0), input_audio.size(1)
            audio_embeddings = self.audio_encoder(
                input_audio.view(-1, *input_audio.shape[2:])
            ).last_hidden_state
            audio_embeddings = audio_embeddings.mean(1)
            audio_embeddings = audio_embeddings.view(
                batch_size, num_samples, *audio_embeddings.shape[1:]
            )
        else:
            audio_embeddings = self.audio_encoder(input_audio)

        fusion_norm = self.dropout(audio_embeddings)

        if self.fusion_head_output_type == "cls":
            x = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            x = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            x = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            x = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        x = self.dropout(x)
        for i in range(len(self.linear_layer_output)):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifer(x)
        return out, x