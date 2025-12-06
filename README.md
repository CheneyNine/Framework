# Multi-Source Traffic Prediction Framework

This is a modular Deep Learning framework built with **PyTorch Lightning** for traffic prediction tasks (e.g., STGCN). It supports multi-source data loading, configurable model selection, and comprehensive logging.

## Project Structure

The project is organized into modular components to ensure scalability and ease of maintenance.

```text
Framework/
├── configs/                 # Configuration files (YAML)
│   ├── global.yaml          # Training & Project settings (Batch size, LR, Seed)
│   └── STGCN.yaml           # Model-specific hyperparameters
│
├── data/                    # Data Loading Layer
│   ├── datamodule.py        # Lightning DataModule (manages Train/Val/Test loaders)
│   ├── dataset.py           # PyTorch Dataset implementation
│   ├── factory.py           # Dataset path configuration & creation logic
│   └── sampler.py           # Custom batch sampler implementation
│
├── models/                  # Model Layer
│   ├── base_model.py        # LightningModule base class (Train/Val/Test steps)
│   ├── builder.py           # Model Factory (initializes models based on config)
│   ├── STGCN.py             # STGCN Network architecture
│   └── callst.py            # GPT-2 + LoRA causal debiasing model
│
├── utils/                   # Utilities & Helper Functions
│   ├── config.py            # Config loading and merging logic
│   ├── metrics.py           # Evaluation metrics (Masked MAE, RMSE, etc.)
│   ├── logging.py           # Python logging setup
│   └── callbacks.py         # Custom Lightning callbacks (TextLogging)
│
└── main.py                  # Application Entry Point
