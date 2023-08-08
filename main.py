from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

import torch
import pandas as pd
import pytorch_lightning as pl

from dataset import BKG, SIG, AtlasDataset
from preprocess import train_val_test_split, normalize
from vae import LitVariationaAutoEncoder


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    bkg = pd.read_hdf("data/bkg_pythia_sanitised_features.h5")
    sig = pd.read_hdf("data/hg3000_hq1000_pythia_sanitised_features.h5")

    train, val, test = train_val_test_split(bkg, sig, sample_size=100_000)

    scaler = StandardScaler()
    x_train, y_train, w_train = normalize(train, scaler, is_train=True)
    x_val, y_val, w_val = normalize(val, scaler)
    x_test, y_test, w_test = normalize(test, scaler)

    vae_train_dataset = AtlasDataset(x_train[y_train == BKG], y_train[y_train == BKG], w_train[y_train == BKG])
    vae_val_dataset = AtlasDataset(x_val[y_val == BKG], y_val[y_val == BKG], w_val[y_val == BKG])
    vae_test_dataset = AtlasDataset(x_test, y_test, w_test)

    vae_train_loader = DataLoader(vae_train_dataset, batch_size=1024, shuffle=True, num_workers=16)
    vae_val_loader = DataLoader(vae_val_dataset, batch_size=1024, shuffle=False, num_workers=16)
    vae_test_loader = DataLoader(vae_test_dataset, batch_size=1024, shuffle=False, num_workers=16)

    vae = LitVariationaAutoEncoder("fcnc", x_train.shape[1], 4, [64, 32, 16, 12], rec_loss="mse", lr_patience=10, rec_loss_weight=1e4)

    vae_trainer = pl.Trainer(
        default_root_dir='logs',
        accelerator='gpu',
        max_epochs=1000,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=25, mode='min', min_delta=0.001, verbose=True),
            pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=True, filename='vae_best'),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        ],
        logger=pl.loggers.TensorBoardLogger('logs/', name='vae'),
        log_every_n_steps=30
    )

    # vae_trainer.test(vae, vae_test_loader)
    # vae_trainer.fit(vae, vae_train_loader, vae_val_loader)
    # vae_trainer.test(vae, vae_test_loader)