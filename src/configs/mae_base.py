from configs.base import Config as BaseConfig
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Config(BaseConfig):
    # Base
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        self.add_args()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_args(self, **kwargs):
        self.batch_size = 1
        self.num_epochs = 20

        self.loss_type = "CrossEntropyLoss"

        self.checkpoint_dir = str(PROJECT_ROOT / "checkpoints" / "IEMOCAP")

        self.model_type = "TriMemoCMT"

        self.text_encoder_type = "bert"  # [bert, roberta]
        self.text_encoder_dim = 768
        self.text_unfreeze = False

        self.audio_encoder_type = "hubert_base"
        self.audio_encoder_dim = 768
        self.audio_unfreeze = False

        self.video_encoder_type: str = "VideoMAEv2"
        self.video_encoder_dim: int = 768
        self.video_unfreeze: bool = True
        self.video_unfreeze_amount: int = 3
        # self.dropout: float = 0.2
        self.fusion_dim: int = 768

        # Dataset
        self.data_name: str = "IEMOCAP"
        self.data_root = str(PROJECT_ROOT / "scripts" / "IEMOCAP_preprocessed")
        self.data_valid: str = "val.pkl"
        self.text_max_length: int = 297
        self.audio_max_length: int = 128000  # 160220
        self.video_max_length: int = 128000
        self.fusion_head_output_type: str = "cls_concat"

        # Config name
        self.name = (
            f"{self.model_type}_{self.text_encoder_type}_{self.audio_encoder_type}_{self.video_encoder_type}"
        )

        for key, value in kwargs.items():
            setattr(self, key, value)
