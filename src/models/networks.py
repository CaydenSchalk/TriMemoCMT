import torch
import torch.nn as nn

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
        text_audio, _ = self.text_cross(audio, text)
        text_video, _ = self.text_cross(video, text)

        # audio attends to text and video
        audio_text, _ = self.audio_cross(text, audio)
        audio_video, _ = self.audio_cross(video, audio)

        # video attends to audio and text
        video_audio, _ = self.video_cross(audio, video)
        video_text, _ = self.video_cross(text, video)

        return text_audio, text_video, audio_text, audio_video, video_audio, video_text

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
        self.video_encoder.to(device)

        # Freeze/Unfreeze the video module
        for param in self.video_encoder.parameters():
            param.requires_grad = cfg.video_unfreeze

        # Fusion module
        self.dropout = nn.Dropout(cfg.dropout)

        self.fusion_module = MultimodalFusion(cfg.text_encoder_dim,
                                              cfg.audio_encoder_dim,
                                              cfg.video_encoder_dim,
                                              cfg.fusion_dim,
                                              cfg.num_attention_head,
                                              cfg.dropout)

        self.linear_layer_output = cfg.linear_layer_output

        previous_dim = cfg.fusion_dim
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
        input_video: torch.Tensor,
        output_attentions: bool = False,
    ):

        text_embeddings = self.text_encoder(input_text).last_hidden_state
        video_embeddings = self.video_encoder(input_video)

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

        # print(f"text: {text_embeddings.shape}, audio: {audio_embeddings.shape}, video: {video_embeddings.shape}")

        ## Fusion Module

        (text_audio,
         text_video,
         audio_text,
         audio_video,
         video_audio,
         video_text) = self.fusion_module(text_embeddings, audio_embeddings, video_embeddings)

        # Concatenate the text and audio embeddings
        fusion_norm = torch.cat((text_audio,
                                 text_video,
                                 audio_text,
                                 audio_video,
                                 video_audio,
                                 video_text), 1)

        fusion_norm = self.dropout(fusion_norm)

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

        # if output_attentions:
        #     return [out, cls_token_final_fusion_norm], [
        #         text_audio_attn_output_weights,
        #         audio_text_attn_output_weights,
        #     ]

        return out, cls_token_final_fusion_norm, text_audio, text_video, audio_text, audio_video, video_audio, video_text

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
