from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
import torch

torch.set_float32_matmul_precision("high")

# Workaround for 'too many open files' error
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

from vq_clip.embedding_dataset import LightningEmbeddingDataModule
from vq_clip.trainer import LightningVQCLIPTrainer


def main():
    cli = LightningCLI(
        LightningVQCLIPTrainer,
        LightningEmbeddingDataModule,
        trainer_defaults={
            "callbacks": [
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(
                    save_last=True,
                ),
            ],
        },
    )


if __name__ in {"__console__", "__main__"}:
    main()
