#!/usr/bin/env python3
import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import MicrographCleaner
from dataset import TrainMicrographDataset, ValidationMicrographDataset
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
def main():
    # Training parameters
    window_size = 512
    batch_size = 8
    n_epochs = 30

    # Load and split data
    train_df = pd.read_csv('train.csv')
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=44)

    # Create datasets and dataloaders
    # 2. Define Datasets
    train_dataset = TrainMicrographDataset(train_df, window_size=window_size)
    val_dataset = ValidationMicrographDataset(val_df, window_size=window_size)

    # 3. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model_self_att = MicrographCleaner(architecture='self-att', learning_rate=1e-5)

    # Setup training
    logger = TensorBoardLogger('lightning_logs', name='micrograph_cleaner')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_iou',
        dirpath='checkpoints',
        filename='micrograph-{epoch:02d}-{val_iou:.2f}',
        save_top_k=3,
        mode='max'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_iou',
        patience=10,
        mode='max',
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices=1,
        precision=32,  # Enables mixed precision
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10,
        gradient_clip_val=1
    )
    # Train model
    trainer.fit(model_self_att, train_loader, val_loader)


if __name__ == "__main__":
    main()