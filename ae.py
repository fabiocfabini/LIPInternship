from datetime import datetime
from typing import Literal, List

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from dataset import BKG, SIG, AtlasDataset
from loss_functions import mse_loss, mae_loss
from preprocess import train_val_test_split, normalize


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, input_dim: int, latent_dim: int, arch: List[int], feature_names: List[str]):
            super().__init__()

            self.input_dim = input_dim
            self.latent_dim = latent_dim
            self.arch = arch
            self.feature_names = feature_names

            self.training_losses = []
            self.validation_losses = []
            self.validation_x = []
            self.validation_x_rec = []

            if len(arch) == 0:
                raise ValueError("Encoder architecture must have at least one layer")

            if latent_dim >= arch[-1]:
                raise ValueError("Latent dimension must be smaller than the last encoder layer")

            self.encoder = self._encoder()
            self.decoder = self._decoder()
            self.is_sanity_check = True

    def _encoder(self) -> nn.Sequential:
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.arch[0]),
            nn.ReLU()
        )

        for i in range(1, len(self.arch)):
            self.encoder.add_module(f"fc({i})", nn.Linear(self.arch[i - 1], self.arch[i]))
            self.encoder.add_module(f"A({i})", nn.ReLU())

        self.encoder.add_module("encoder_output", nn.Linear(self.arch[-1], self.latent_dim))
        self.encoder.add_module(f"A({i+1})", nn.ReLU())

        return self.encoder

    def _decoder(self) -> nn.Sequential:
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.arch[-1]),
            nn.ReLU()
        )

        for i in range(len(self.arch) - 1, 0, -1):
            self.decoder.add_module(f"fc({i})", nn.Linear(self.arch[i], self.arch[i - 1]))
            self.decoder.add_module(f"A({i})", nn.ReLU())

        self.decoder.add_module("decoder_output", nn.Linear(self.arch[0], self.input_dim))

        return self.decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        X, _, w = batch

        X_preds = self(X)
        loss = torch.einsum('ij,i->ij', torch.abs(X_preds - X), w).sum()
        self.training_losses.append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        loss = torch.stack(self.training_losses)
        self.log('training_loss', loss.mean())
        print(len(self.training_losses))
        self.training_losses.clear()

    def validation_step(self, batch, batch_idx):
        X, _, w = batch

        X_preds = self(X)
        loss = torch.einsum('ij,i->ij', torch.abs(X_preds - X), w).sum()
        self.validation_losses.append(loss)
        self.validation_x.append(X)
        self.validation_x_rec.append(X_preds)
        return loss

    def log_reconstruction(self):
        x = torch.cat(self.validation_x).cpu().detach().numpy()
        x_rec = torch.cat(self.validation_x_rec).cpu().detach().numpy()

        df_x = pd.DataFrame(x, columns=self.feature_names)
        df_x_rec = pd.DataFrame(x_rec, columns=self.feature_names)

        # log the histograms of the original and reconstructed data
        # original should be a line, reconstructed should be a filled area
        for col in df_x.columns:
            fig, ax = plt.subplots()
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.hist(df_x[col], bins=100, density=True, histtype='step', label='original')
            ax.hist(df_x_rec[col], bins=100, density=True, label='reconstructed', alpha=0.5)
            ax.set_yscale('log')
            ax.legend()
            self.logger.experiment.add_figure(f'histograms/{col}', fig, self.current_epoch)


    def on_validation_epoch_end(self) -> None:
        if not self.is_sanity_check:
            loss = torch.stack(self.validation_losses)
            self.log('val_loss', loss.mean())
            self.log_reconstruction()

        self.is_sanity_check = False
        self.validation_losses.clear()
        self.validation_x.clear()
        self.validation_x_rec.clear()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    bkg = pd.read_hdf("data/bkg_pythia_sanitised_features.h5")
    sig = pd.read_hdf("data/fcnc_pythia_sanitised_features.h5")

    train, val, test = train_val_test_split(bkg, sig, sample_size=100_000)

    scaler = StandardScaler()
    x_train, y_train, w_train = normalize(train, scaler, is_train=True)
    x_val, y_val, w_val = normalize(val, scaler)
    x_test, y_test, w_test = normalize(test, scaler)

    vae_train_dataset = AtlasDataset(x_train[y_train == BKG], y_train[y_train == BKG], w_train[y_train == BKG])
    vae_val_dataset = AtlasDataset(x_val[y_val == BKG], y_val[y_val == BKG], w_val[y_val == BKG])
    vae_test_dataset = AtlasDataset(x_test, y_test, w_test)

    vae_train_loader = DataLoader(vae_train_dataset, batch_size=512, shuffle=True, num_workers=16)
    vae_val_loader = DataLoader(vae_val_dataset, batch_size=512, shuffle=False, num_workers=16)
    vae_test_loader = DataLoader(vae_test_dataset, batch_size=512, shuffle=False, num_workers=16)

    vae = LitAutoEncoder(x_train.shape[1], 4, [64, 32, 16, 12], feature_names=[fet for fet in train.columns if fet not in ['is_signal', 'weight']])

    vae_trainer = pl.Trainer(
        default_root_dir='logs',
        accelerator='gpu',
        max_epochs=1000,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=25, mode='min', min_delta=0.001, verbose=True),
            pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=True, filename='vae_best'),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        ],
        logger=pl.loggers.TensorBoardLogger('logs/', name='ae'),
        log_every_n_steps=30
    )

    vae_trainer.fit(vae, vae_train_loader, vae_val_loader)
