import torch.nn as nn
import torchaudio
from transformers import (
    BertConfig,
    BertModel,
    RobertaConfig,
    RobertaModel,
    FocalNetConfig,
    FocalNetModel,
    VideoMAEImageProcessor,
    AutoModel,
    AutoConfig
)

from configs.base import Config

class MAEBase(nn.Module):
    def __init__(self, **kwargs):
        super(MAEBase, self).__init__(**kwargs)
        self.config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True)
        self.processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Base")
        self.model = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Base', config=self.config, trust_remote_code=True)

        # trying to get how VideoMAE v2's forward func works
        # I'm trying to avoid the pooling so I can get the dimensions right

        # print(type(self.model))
        # print([n for n, _ in self.model.model.named_children()])
        # import inspect
        # print(inspect.getsource(self.model.model.forward))
        # print(inspect.getsource(self.model.model.forward_features))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        inner = self.model.model
        B = x.size(0)

        x = inner.patch_embed(x)
        if inner.pos_embed is not None:
            x = x + inner.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = inner.pos_drop(x)
        for blk in inner.blocks:
            x = blk(x)
        x = inner.norm(x)  # (B, num_patches, 768)
        return x

def build_mae_encoder(cfg : Config) -> nn.Module:
    """A function to build the VideoMAEv2 encoder"""
    return MAEBase()

def build_bert_encoder() -> nn.Module:
    """A function to build bert encoder"""
    config = BertConfig.from_pretrained(
        "bert-base-uncased", output_hidden_states=True, output_attentions=True
    )
    bert = BertModel.from_pretrained("bert-base-uncased", config=config)
    return bert


class HuBertBase(nn.Module):
    def __init__(self, **kwargs):
        super(HuBertBase, self).__init__(**kwargs)
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model()

    def forward(self, x):
        features, _ = self.model(x)
        return features


def build_hubert_base_encoder(cfg: Config) -> nn.Module:
    """A function to build hubert encoder"""
    return HuBertBase()


def build_audio_encoder(cfg: Config) -> nn.Module:
    """A function to build audio encoder

    Args:
        cfg (Config): Config object

    Returns:
        nn.Module: Audio encoder
    """
    type = cfg.audio_encoder_type

    encoders = {
        "hubert_base": build_hubert_base_encoder,
    }
    assert type in encoders.keys(), f"Invalid audio encoder type: {type}"
    return encoders[type](cfg)


def build_text_encoder(type: str = "bert") -> nn.Module:
    """A function to build text encoder

    Args:
        type (str, optional): Type of text encoder. Defaults to "bert".

    Returns:
        torch.nn.Module: Text encoder
    """
    encoders = {
        "bert": build_bert_encoder,
    }
    assert type in encoders.keys(), f"Invalid text encoder type: {type}"
    return encoders[type]()

def build_video_encoder(cfg: Config) -> nn.Module:
    # A function to build the video encoder

    type = cfg.video_encoder_type

    encoders = {
        "VideoMAEv2": build_mae_encoder,
    }
    assert type in encoders.keys(), f"Invalid video encoder type: {type}"
    return encoders[type](cfg)