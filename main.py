import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import swanlab
from utils.swanlab_logger import SwanLabLogger
import argparse
import os
import torch
import logging

from data import MultiDataModule, get_dataset_paths
from models import get_model_instance
from utils import setup_text_logging, TextLoggingCallback, load_config

def main(args):
    global_config_path = f"{args.config}"
    model_config_path = f"{args.model_config}"

    if not os.path.exists(global_config_path):
        raise FileNotFoundError(f"Global config file not found: {global_config_path}")
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config file not found: {model_config_path}")

    config = load_config(global_config_path, model_config_path)

    L.seed_everything(config['project']['seed'])
    torch.set_float32_matmul_precision(config['project']['matmul_precision'])

    model_name = config['model']['name']
    logger_name = f"{model_name}_log"
    csv_logger = CSVLogger('logs/', name=logger_name)
    log_dir = csv_logger.log_dir

    swan_config = {
        k: (str(v) if isinstance(v, torch.device) else v)
        for k, v in vars(args).items()
    }
    swan_config.update(config)

    loggers = [csv_logger]
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        swanlab_logger = SwanLabLogger(
            workspace="st2025",
            project="Causal_Analysis",
            experiment_name=model_name+"_developing",
            config=swan_config,
        )
        loggers.append(swanlab_logger)

        setup_text_logging(log_dir)
        logging.info(f"----- Experiment Configuration -----")
        logging.info(f"Model: {model_name}")
        logging.info(f"Global Config: {global_config_path}")
        logging.info(f"Model Config: {model_config_path}")
        logging.info(f"GPU ID(s): {config['project']['gpus']}")

    dataset_configs = []
    for name in config['data']['datasets']:
        ds_conf = get_dataset_paths(name, root_dir=config['data']['root_dir'])
        dataset_configs.append(ds_conf)

    data_module = MultiDataModule(
        dataset_configs=dataset_configs,
        batch_size=config['data']['batch_size'],
        history_seq_len=config['data']['history_seq_len'],
        future_seq_len=config['data']['future_seq_len'],
        stride=config['data']['stride'],
        num_workers=config['data']['num_workers']
    )

    model = get_model_instance(config, data_module)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(log_dir, 'checkpoints'),
        filename=f'{model_name}-{{epoch:02d}}-{{val_loss:.4f}}',
        save_top_k=1,
        mode='min',
    )
    early_stop_callback = EarlyStopping(
        monitor=config['train']['scheduler']['monitor'],
        patience=config['train']['patience'],
        verbose=False,
        mode='min'
    )
    text_log_callback = TextLoggingCallback()

    devices = [int(x) for x in str(config['project']['gpus']).split(',')]
    strategy = 'ddp' if len(devices) > 1 else 'auto'

    trainer = L.Trainer(
        max_epochs=config['train']['max_epochs'],
        accelerator=config['project']['accelerator'],
        devices=devices,
        strategy=strategy,
        callbacks=[checkpoint_callback, early_stop_callback, text_log_callback],
        logger=loggers,
        enable_progress_bar=True,
        use_distributed_sampler=False
    )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logging.info("----- Starting Training -----")

    trainer.fit(model, datamodule=data_module)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        logging.info("----- Running Test on the Best Model -----")

    test_results = trainer.test(model, datamodule=data_module, ckpt_path='best', verbose=False)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        for i, result in enumerate(test_results):
            dataset_name = data_module.hparams.dataset_configs[i].get('name', f'Dataset_{i}')
            logging.info(f"--- Results for {dataset_name} ---")
            for key, value in result.items():
                if f"/{dataset_name}/" in key:
                    metric_name = key.split('/')[-1]
                    logging.info(f"  {metric_name}: {value:.6f}")

        logging.info(f"All Done. Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Lightning Multi-Source Traffic Prediction")
    parser.add_argument('--config', type=str, required=True, help='Normal config to run ')
    parser.add_argument('--model_config', type=str, required=True, help='Model config to run')
    args = parser.parse_args()
    main(args)