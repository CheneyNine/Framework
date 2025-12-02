import logging
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
import torch

class TextLoggingCallback(Callback):
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        train_loss = metrics.get('train_loss_epoch', metrics.get('train_loss'))
        if train_loss is not None:
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.item()
            logging.info(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.6f}")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        val_loss = metrics.get('val_loss')
        if val_loss is not None:
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.item()
            logging.info(f"[Epoch {epoch:03d}] Valid Loss (Avg): {val_loss:.6f}")