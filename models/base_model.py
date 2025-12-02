import lightning as L
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import masked_mae, masked_rmse


class BaseModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self._validation_outputs_per_dataloader = []
        self._test_outputs_per_dataloader = []

    def on_validation_start(self) -> None:
        num_dataloaders = len(self.trainer.datamodule.val_dataloader())
        self._validation_outputs_per_dataloader = [[] for _ in range(num_dataloaders)]

    def on_test_start(self) -> None:
        num_dataloaders = len(self.trainer.datamodule.test_dataloader())
        self._test_outputs_per_dataloader = [[] for _ in range(num_dataloaders)]

    def configure_optimizers(self):
        optimizer_config = self.config['train']['optimizer']
        scheduler_config = self.config['train']['scheduler']
        opt_name = optimizer_config.get('name', 'Adam')

        if opt_name == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        elif opt_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        if scheduler_config and scheduler_config['name'] == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                'min',
                patience=scheduler_config['patience']
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scheduler_config['monitor']
                }
            }
        return optimizer

    def forward(self, batch):
        raise NotImplementedError("Each model must implement its own forward pass.")

    def _get_reals_and_preds(self, batch):
        x, y, adj = batch[:3]
        y_hat = self.forward(x, adj)
        return y, y_hat

    def _unscale_data(self, data: torch.Tensor, scaler):
        original_shape = data.shape
        device = data.device
        data_np = data.cpu().numpy().squeeze()
        data_reshaped = data_np.transpose(0, 2, 1).reshape(-1, original_shape[1])
        unscaled_data_reshaped = scaler.inverse_transform(data_reshaped)
        unscaled_data_np = unscaled_data_reshaped.reshape(original_shape[0], original_shape[-1], original_shape[1])
        unscaled_data_np = unscaled_data_np.transpose(0, 2, 1)
        return torch.from_numpy(unscaled_data_np).unsqueeze(2).to(device)

    def training_step(self, batch, batch_idx):
        current_batch_size = batch[0].size(0)
        y, y_hat = self._get_reals_and_preds(batch)
        loss = masked_mae(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=current_batch_size)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        y, y_hat = self._get_reals_and_preds(batch)
        scaler = self.trainer.datamodule.scalers[dataloader_idx]
        y_unscaled = self._unscale_data(y, scaler)
        y_hat_unscaled = self._unscale_data(y_hat, scaler)

        metrics = {
            'val_loss': masked_mae(y_hat, y),
        }
        self._validation_outputs_per_dataloader[dataloader_idx].append(metrics)

    def on_validation_epoch_end(self):
        all_dataloader_losses = []
        for i, outputs in enumerate(self._validation_outputs_per_dataloader):
            if not outputs:
                continue
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            all_dataloader_losses.append(avg_loss)

        if all_dataloader_losses:
            global_avg_loss = torch.stack(all_dataloader_losses).mean()
            self.log('val_loss', global_avg_loss, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        y, y_hat = self._get_reals_and_preds(batch)
        scaler = self.trainer.datamodule.scalers[dataloader_idx]
        y_unscaled = self._unscale_data(y, scaler)
        y_hat_unscaled = self._unscale_data(y_hat, scaler)

        metrics = {
            'loss': masked_mae(y_hat, y),
            'mae': masked_mae(y_hat_unscaled, y_unscaled),
            'rmse': masked_rmse(y_hat_unscaled, y_unscaled)
        }
        self._test_outputs_per_dataloader[dataloader_idx].append(metrics)

    def on_test_epoch_end(self):
        datasets = self.trainer.datamodule.dataset_configs
        for i, outputs in enumerate(self._test_outputs_per_dataloader):
            if not outputs:
                continue

            dataset_name = datasets[i].get('name', f'Dataset_{i}')

            avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
            avg_mae = torch.stack([x['mae'] for x in outputs]).mean()
            avg_rmse = torch.stack([x['rmse'] for x in outputs]).mean()

            metrics_dict = {
                f'test/{dataset_name}/loss': avg_loss,
                f'test/{dataset_name}/mae': avg_mae,
                f'test/{dataset_name}/rmse': avg_rmse,
            }
            self.log_dict(metrics_dict, sync_dist=True)