import torch
import torch.nn as nn

from configs.base import Config

from .modules import build_audio_encoder, build_text_encoder, build_video_encoder
from torch.utils.checkpoint import checkpoint

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

class TemporalTransformer(nn.Module):
    """Self-attention over conversation turns."""
    def __init__(self, embed_dim, num_heads, num_layers=2, dropout=0.1):
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, embed_dim) * 0.02)
        self.speaker_embed = nn.Embedding(2, embed_dim)  # M=0, F=1
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, speaker_ids=None, mask=None):
        """
        x: (B, T_conv, D)
        speaker_ids: (B, T_conv) — 0 or 1
        mask: (B, T_conv) — True for padded positions
        """
        T = x.size(1)
        x = x + self.pos_encoding[:, :T, :]

        if speaker_ids is not None:
            x = x + self.speaker_embed(speaker_ids)

        # TransformerEncoder expects src_key_padding_mask where True = ignore
        x = self.encoder(x, src_key_padding_mask=mask)
        return x

class SpeakerStateRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_speakers=2, dropout=0.1):
        super().__init__()
        self.global_gru = nn.GRUCell(embed_dim, hidden_dim)
        self.speaker_gru = nn.GRUCell(embed_dim + hidden_dim, hidden_dim)
        self.emotion_gru = nn.GRUCell(embed_dim + hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, utt_embeds, speaker_ids):
        # utt_embeds: (B, T_conv, D)
        # speaker_ids: (B, T_conv)
        B, T_conv, D = utt_embeds.shape

        global_state = torch.zeros(B, self.hidden_dim).to(utt_embeds.device)
        speaker_states = torch.zeros(B, 2, self.hidden_dim).to(utt_embeds.device)

        outputs = []
        for t in range(T_conv):
            utt = utt_embeds[:, t, :]           # (B, D)
            spk = speaker_ids[:, t]              # (B,)

            # Update global state
            global_state = self.global_gru(utt, global_state)

            # Get current speaker's state
            spk_state = speaker_states[torch.arange(B), spk]  # (B, H)

            # Update speaker state
            new_spk_state = self.speaker_gru(
                torch.cat([utt, global_state], dim=-1), spk_state
            )
            speaker_states[torch.arange(B), spk] = new_spk_state

            # Emotion state
            emotion = self.emotion_gru(
                torch.cat([utt, new_spk_state], dim=-1),
                torch.zeros_like(new_spk_state)
            )
            outputs.append(emotion)

        return torch.stack(outputs, dim=1)  # (B, T_conv, H)

class TriMemoCMT(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(TriMemoCMT, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)

        # Freeze/Unfreeze the text module
        for param in self.text_encoder.parameters():
            param.requires_grad = cfg.text_unfreeze

        # Audio module
        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)

        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = cfg.audio_unfreeze

        # Video module
        self.video_encoder = build_video_encoder(cfg)

        # for name, _ in self.video_encoder.named_modules():
        #     print(name)

        self.video_encoder.to(device)

        # Freeze/Unfreeze the video module
        for param in self.video_encoder.parameters():
            param.requires_grad = False

        if cfg.video_unfreeze:
            for block in self.video_encoder.model.model.blocks[-cfg.video_unfreeze_amount:]:
                for p in block.parameters():
                    p.requires_grad = True

        # for name, param in self.video_encoder.named_parameters():
        #     print(name, param.shape)

        # Fusion module
        self.dropout = nn.Dropout(cfg.dropout)

        self.fusion_module = MultimodalFusion(cfg.text_encoder_dim,
                                              cfg.audio_encoder_dim,
                                              cfg.video_encoder_dim,
                                              cfg.fusion_dim,
                                              cfg.num_attention_head,
                                              cfg.dropout)

        self.linear_layer_output = cfg.linear_layer_output

        self.fusion_head_output_type = cfg.fusion_head_output_type

        previous_dim = cfg.fusion_dim

        if cfg.fusion_head_output_type == "cls_concat":
            previous_dim = 6 * cfg.fusion_dim

        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type
        self.video_cls_token = nn.Parameter(torch.randn(1, 1, cfg.video_encoder_dim))

        self.video_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=cfg.video_encoder_dim, nhead=cfg.num_attention_head, dropout=cfg.dropout,
                                       batch_first=True),
            num_layers=2
        )

        temporal_input_dim = (6 * cfg.fusion_dim
                              if cfg.fusion_head_output_type == "cls_concat"
                              else cfg.fusion_dim)

        self.temporal_transformer = TemporalTransformer(
            embed_dim=temporal_input_dim,
            num_heads=cfg.num_attention_head,
            num_layers=cfg.temporal_num_layers,
            dropout=cfg.dropout,
        )

        self.use_temporal = cfg.use_temporal

    # def forward(
    #     self,
    #     input_text: torch.Tensor,
    #     input_audio: torch.Tensor,
    #     input_video: torch.Tensor,
    #     output_attentions: bool = False,
    # ):
    #
    #     text_embeddings = self.text_encoder(input_text).last_hidden_state
    #
    #     # add a CLS token at the beginning
    #     video_embeddings = self.video_encoder(input_video)  # (B, num_patches, 768)
    #
    #     # cls = self.video_cls_token.expand(video_embeddings.size(0), -1, -1)  # (B, 1, 768)
    #     # video_embeddings = torch.cat([cls, video_embeddings], dim=1)  # (B, 1+num_patches, 768)
    #
    #     cls = self.video_cls_token.expand(video_embeddings.size(0), -1, -1)
    #     video_embeddings = torch.cat([cls, video_embeddings], dim=1)  # (B, N+1, 768)
    #     video_embeddings = self.video_self_attn(video_embeddings)
    #
    #     if len(input_audio.size()) != 2:
    #         batch_size, num_samples = input_audio.size(0), input_audio.size(1)
    #         audio_embeddings = self.audio_encoder(
    #             input_audio.view(-1, *input_audio.shape[2:])
    #         ).last_hidden_state
    #         audio_embeddings = audio_embeddings.mean(1)
    #         audio_embeddings = audio_embeddings.view(
    #             batch_size, num_samples, *audio_embeddings.shape[1:]
    #         )
    #     else:
    #         audio_embeddings = self.audio_encoder(input_audio)
    #
    #     # print(f"text: {text_embeddings.shape}, audio: {audio_embeddings.shape}, video: {video_embeddings.shape}")
    #
    #     ## Fusion Module
    #
    #     (text_audio,
    #      text_video,
    #      audio_text,
    #      audio_video,
    #      video_audio,
    #      video_text) = self.fusion_module(text_embeddings, audio_embeddings, video_embeddings)
    #
    #     # Concatenate the text and audio embeddings
    #     fusion_norm = torch.cat((text_audio,
    #                              text_video,
    #                              audio_text,
    #                              audio_video,
    #                              video_audio,
    #                              video_text), 1)
    #
    #     fusion_norm = self.dropout(fusion_norm)
    #
    #     cls_tokens = torch.stack([
    #         text_audio[:, 0, :],
    #         text_video[:, 0, :],
    #         audio_text[:, 0, :],
    #         audio_video[:, 0, :],
    #         video_audio[:, 0, :],
    #         video_text[:, 0, :],
    #     ], dim=1)  # [B, 6, hidden]
    #
    #     B = text_audio.size(0)
    #
    #     num_cross = cls_tokens.size(1)
    #     hidden_size = cls_tokens.size(2)
    #
    #     # Get classification output
    #     if self.fusion_head_output_type == "cls_concat":
    #         cls_token_final_fusion_norm = cls_tokens.reshape(B, num_cross * hidden_size)
    #     elif self.fusion_head_output_type == "cls_mean":
    #         cls_token_final_fusion_norm = cls_tokens.mean(dim=1)  # [B, hidden]
    #     elif self.fusion_head_output_type == "cls":
    #         cls_token_final_fusion_norm = fusion_norm[:, 0, :]
    #     elif self.fusion_head_output_type == "mean":
    #         cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
    #     elif self.fusion_head_output_type == "max":
    #         cls_token_final_fusion_norm = fusion_norm.max(dim=1)[0]
    #     elif self.fusion_head_output_type == "min":
    #         cls_token_final_fusion_norm = fusion_norm.min(dim=1)[0]
    #     else:
    #         raise ValueError("Invalid fusion head output type")
    #
    #     # Classification head
    #     x = cls_token_final_fusion_norm
    #     x = self.dropout(x)
    #
    #     # print(x.shape)
    #     # print(fusion_norm[:, 0, :].shape)
    #     for i, _ in enumerate(self.linear_layer_output):
    #         x = getattr(self, f"linear_{i}")(x)
    #         x = nn.functional.leaky_relu(x)
    #     x = self.dropout(x)
    #     out = self.classifer(x)
    #
    #     # if output_attentions:
    #     #     return [out, cls_token_final_fusion_norm], [
    #     #         text_audio_attn_output_weights,
    #     #         audio_text_attn_output_weights,
    #     #     ]
    #
    #     return out, cls_token_final_fusion_norm, text_audio, text_video, audio_text, audio_video, video_audio, video_text

    def _encode_chunked(self, text_flat, audio_flat, video_flat, chunk_size=4):
        N = text_flat.size(0)
        text_embeds, audio_embeds, video_embeds = [], [], []

        for i in range(0, N, chunk_size):
            j = min(i + chunk_size, N)

            with torch.no_grad():
                t = self.text_encoder(text_flat[i:j]).last_hidden_state
                a = self.audio_encoder(audio_flat[i:j])

            # Gradient checkpointing: recomputes forward during backward
            # instead of storing all intermediate activations
            v = checkpoint(self._encode_single_video, video_flat[i:j], use_reentrant=False)

            text_embeds.append(t)
            audio_embeds.append(a)
            video_embeds.append(v)

        return (torch.cat(text_embeds, dim=0),
                torch.cat(audio_embeds, dim=0),
                torch.cat(video_embeds, dim=0))

    def _encode_single_video(self, video_chunk):
        v = self.video_encoder(video_chunk)
        cls = self.video_cls_token.expand(v.size(0), -1, -1)
        v = torch.cat([cls, v], dim=1)
        v = self.video_self_attn(v)
        return v

    def forward(self, utt_text, utt_audio, utt_video,
                speaker_id, global_state, speaker_states):

        # encode + fuse single utterance (no T_conv dimension)
        text_emb, audio_emb, video_emb = self._encode_chunked(...)
        text_audio, text_video, audio_text, audio_video, video_audio, video_text = \
            self.fusion_module(text_emb, audio_emb, video_emb)

        cls_tokens = torch.stack([
            text_audio[:, 0, :], text_video[:, 0, :],
            audio_text[:, 0, :], audio_video[:, 0, :],
            video_audio[:, 0, :], video_text[:, 0, :],
        ], dim=1)  # (B*T_conv, 6, D)

        if self.fusion_head_output_type == "cls_concat":
            utt_embeds = cls_tokens.reshape(B * T_conv, -1)
        else:  # cls_mean
            utt_embeds = cls_tokens.mean(dim=1)  # (B*T_conv, D)

        # # Reshape back to conversation structure
        # utt_embeds = utt_embeds.view(B, T_conv, -1)  # (B, T_conv, D)

        # one RNN step
        global_state = self.global_gru(utt_embed, global_state)
        spk_state = speaker_states[torch.arange(B), speaker_id]
        new_spk_state = self.speaker_gru(
            torch.cat([utt_embed, global_state], dim=-1), spk_state
        )
        speaker_states[torch.arange(B), speaker_id] = new_spk_state
        emotion = self.emotion_gru(
            torch.cat([utt_embed, new_spk_state], dim=-1),
            torch.zeros_like(new_spk_state)
        )

        out = self.classifer(emotion)
        return out, global_state, speaker_states

    # using temporal transformer
    # def forward(self, input_text, input_audio, input_video, speaker_ids=None, padding_mask=None):
    #     B, T_conv = input_text.shape[:2]
    #
    #     # Always flatten for per-utterance encoding
    #     text_flat  = input_text.view(B * T_conv, *input_text.shape[2:])
    #     audio_flat = input_audio.view(B * T_conv, *input_audio.shape[2:])
    #     video_flat = input_video.view(B * T_conv, *input_video.shape[2:])
    #
    #     # Encode
    #     text_emb, audio_emb, video_emb = self._encode_chunked(
    #         text_flat, audio_flat, video_flat, chunk_size=2
    #     )
    #
    #     # Fuse
    #     text_audio, text_video, audio_text, audio_video, video_audio, video_text = \
    #         self.fusion_module(text_emb, audio_emb, video_emb)
    #
    #     cls_tokens = torch.stack([
    #         text_audio[:, 0, :], text_video[:, 0, :],
    #         audio_text[:, 0, :], audio_video[:, 0, :],
    #         video_audio[:, 0, :], video_text[:, 0, :],
    #     ], dim=1)  # (B*T_conv, 6, D)
    #
    #     if self.fusion_head_output_type == "cls_concat":
    #         utt_embeds = cls_tokens.reshape(B * T_conv, -1)
    #     else:  # cls_mean
    #         utt_embeds = cls_tokens.mean(dim=1)  # (B*T_conv, D)
    #
    #     # Reshape back to conversation structure
    #     utt_embeds = utt_embeds.view(B, T_conv, -1)  # (B, T_conv, D)
    #
    #     # temoral
    #     if self.use_temporal:
    #         x = self.temporal_transformer(
    #             utt_embeds,
    #             speaker_ids=speaker_ids,
    #             mask=padding_mask,
    #         )  # (B, T_conv, D)
    #     else:
    #         x = utt_embeds  # skip temporal, pass through directly
    #
    #     # Classification head
    #     x = self.dropout(x)
    #     for i in range(len(self.linear_layer_output)):
    #         x = getattr(self, f"linear_{i}")(x)
    #         x = nn.functional.leaky_relu(x)
    #     x = self.dropout(x)
    #     out = self.classifer(x)  # (B, T_conv, num_classes)
    #
    #     return out, x

    # def forward(
    #         self,
    #         input_text,  # (B, T_conv, seq_len)
    #         input_audio,  # (B, T_conv, audio_len)
    #         input_video,  # (B, T_conv, num_frames, C, H, W)
    #         speaker_ids=None,  # (B, T_conv)
    #         padding_mask=None,  # (B, T_conv) — True where padded
    # ):
    #     B, T_conv = input_text.shape[:2]
    #
    #     # Flatten conversation into batch for per-utterance encoding
    #     text_flat = input_text.view(B * T_conv, *input_text.shape[2:])
    #     audio_flat = input_audio.view(B * T_conv, *input_audio.shape[2:])
    #     video_flat = input_video.view(B * T_conv, *input_video.shape[2:])
    #
    #     # Per-utterance encoding
    #     # text_embeddings = self.text_encoder(text_flat).last_hidden_state
    #     # video_embeddings = self.video_encoder(video_flat)
    #     # cls = self.video_cls_token.expand(video_embeddings.size(0), -1, -1)
    #     # video_embeddings = torch.cat([cls, video_embeddings], dim=1)
    #     # video_embeddings = self.video_self_attn(video_embeddings)
    #     # audio_embeddings = self.audio_encoder(audio_flat)
    #
    #     text_embeddings, audio_embeddings, video_embeddings = \
    #         self._encode_chunked(text_flat, audio_flat, video_flat, chunk_size=2)
    #
    #     # Fusion
    #     (text_audio, text_video, audio_text,
    #      audio_video, video_audio, video_text) = self.fusion_module(
    #         text_embeddings, audio_embeddings, video_embeddings
    #     )
    #
    #     # Pool to per-utterance vector
    #     cls_tokens = torch.stack([
    #         text_audio[:, 0, :], text_video[:, 0, :],
    #         audio_text[:, 0, :], audio_video[:, 0, :],
    #         video_audio[:, 0, :], video_text[:, 0, :],
    #     ], dim=1)  # (B*T_conv, 6, D)
    #
    #     if self.fusion_head_output_type == "cls_concat":
    #         utt_embeds = cls_tokens.reshape(B * T_conv, -1)  # (B*T_conv, 6*D)
    #     elif self.fusion_head_output_type == "cls_mean":
    #         utt_embeds = cls_tokens.mean(dim=1)  # (B*T_conv, D)
    #
    #     # Reshape back to conversations
    #     utt_embeds = utt_embeds.view(B, T_conv, -1)  # (B, T_conv, D)
    #
    #     temporal_out = utt_embeds
    #
    #     # Temporal transformer
    #     if self.use_temporal:
    #         temporal_out = self.temporal_transformer(
    #             utt_embeds,
    #             speaker_ids=speaker_ids,
    #             mask=padding_mask,
    #         )  # (B, T_conv, D)
    #
    #     # Per-utterance classification
    #     x = self.dropout(temporal_out)
    #     for i, _ in enumerate(self.linear_layer_output):
    #         x = getattr(self, f"linear_{i}")(x)
    #         x = nn.functional.leaky_relu(x)
    #     x = self.dropout(x)
    #     out = self.classifer(x)  # (B, T_conv, num_classes)
    #
    #     return out, temporal_out

    def encode_audio(self, audio: torch.Tensor):
        return self.audio_encoder(audio)

    def encode_text(self, input_ids: torch.Tensor):
        return self.text_encoder(input_ids).last_hidden_state

    def encode_video(self, input_ids: torch.Tensor):
        return self.video_encoder(input_ids)


class TextOnly(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(TextOnly, self).__init__()
        # Text module
        self.text_encoder = build_text_encoder(cfg.text_encoder_type)
        self.text_encoder.to(device)
        # Freeze/Unfreeze the text module
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

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        output_attentions: bool = False,
    ):

        text_embeddings = self.text_encoder(input_text).last_hidden_state
        fusion_norm = self.dropout(text_embeddings)

        # Get classification output
        if self.fusion_head_output_type == "cls":
            cls_token_final_fusion_norm = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            cls_token_final_fusion_norm = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head
        x = cls_token_final_fusion_norm
        x = self.dropout(x)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifer(x)

        return out, cls_token_final_fusion_norm


class AudioOnly(nn.Module):
    def __init__(
        self,
        cfg: Config,
        device: str = "cpu",
    ):
        super(AudioOnly, self).__init__()

        # Audio module
        self.audio_encoder = build_audio_encoder(cfg)
        self.audio_encoder.to(device)

        # Freeze/Unfreeze the audio module
        for param in self.audio_encoder.parameters():
            param.requires_grad = cfg.audio_unfreeze

        self.linear_layer_output = cfg.linear_layer_output

        self.dropout = nn.Dropout(cfg.dropout)

        previous_dim = cfg.audio_encoder_dim
        if len(cfg.linear_layer_output) > 0:
            for i, linear_layer in enumerate(cfg.linear_layer_output):
                setattr(self, f"linear_{i}", nn.Linear(previous_dim, linear_layer))
                previous_dim = linear_layer

        self.classifer = nn.Linear(previous_dim, cfg.num_classes)

        self.fusion_head_output_type = cfg.fusion_head_output_type

    def forward(
        self,
        input_text: torch.Tensor,
        input_audio: torch.Tensor,
        output_attentions: bool = False,
    ):

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

        # Get classification output
        if self.fusion_head_output_type == "cls":
            cls_token_final_fusion_norm = fusion_norm[:, 0, :]
        elif self.fusion_head_output_type == "mean":
            cls_token_final_fusion_norm = fusion_norm.mean(dim=1)
        elif self.fusion_head_output_type == "max":
            cls_token_final_fusion_norm = fusion_norm.max(dim=1)[0]
        elif self.fusion_head_output_type == "min":
            cls_token_final_fusion_norm = fusion_norm.min(dim=1)[0]
        else:
            raise ValueError("Invalid fusion head output type")

        # Classification head
        x = cls_token_final_fusion_norm
        x = self.dropout(x)
        for i, _ in enumerate(self.linear_layer_output):
            x = getattr(self, f"linear_{i}")(x)
            x = nn.functional.leaky_relu(x)
        x = self.dropout(x)
        out = self.classifer(x)

        return out, cls_token_final_fusion_norm
