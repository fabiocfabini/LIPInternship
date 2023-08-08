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


class LitVariationaAutoEncoder(pl.LightningModule):
    def __init__(self,
                save_as: str,
                input_dim: int,
                latent_dim: int, 
                arch: List[int], 
                features: List[str],
                rec_loss: Literal["mse", "mae"] = "mse", 
                rec_loss_weight: float = 1.0, 
                kl_loss_weight: float = 1.0, 
                lr_patience: int = 5,
        ):
        super().__init__()
        self.save_as = save_as

        self.arch = arch
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.features = features

        self.encoder = self._encoder()
        self.decoder = self._decoder()

        if rec_loss == "mse":
            self.rec_loss = mse_loss
        elif rec_loss == "mae":
            self.rec_loss = mae_loss
        else:
            raise ValueError("rec_loss must be one of ['mse', 'mae']")
        self.rec_loss_weight = rec_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.lr_patience = lr_patience

        self.train_rec_losses = []
        self.train_kl_losses = []
        self.val_losses = []
        self.val_x = []
        self.val_x_rec = []
        self.test_losses = []
        self.test_labels = []
        self.is_sanity_check = True

    def _encoder(self) -> nn.Sequential:
        """The encoder recieves [input_dim] features as input 
        and outputs [latent_dim * 2] features. It must 
        have the [arch] architecture.

        Returns:
            nn.Sequential: The encoder
        """

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.arch[0]),
            nn.LeakyReLU()
        )

        for i in range(1, len(self.arch)):
            self.encoder.add_module(f"fc({i})", nn.Linear(self.arch[i - 1], self.arch[i]))
            self.encoder.add_module(f"A({i})", nn.LeakyReLU())

        self.encoder.add_module(f"A({i+1})", nn.LeakyReLU())
        self.encoder.add_module("mu+sigma", nn.Linear(self.arch[-1], self.latent_dim * 2))

        return self.encoder

    def _decoder(self) -> nn.Sequential:
        """The decoder recieves [latent_dim] features
        as input and outputs [input_dim] features. It must have the
        [arch] architecture.

        Returns:
            nn.Sequential: The decoder
        """

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.arch[-1]),
            nn.LeakyReLU()
        )

        for i in range(len(self.arch) - 1, 0, -1):
            self.decoder.add_module(f"fc({i})", nn.Linear(self.arch[i], self.arch[i - 1]))
            self.decoder.add_module(f"A({i})", nn.LeakyReLU())

        self.decoder.add_module("decoder_output", nn.Linear(self.arch[0], self.input_dim))

        return self.decoder

    def _get_mu(self, mu_sigma: torch.Tensor) -> torch.Tensor:
        """Given the output of the encoder, returns the mu vector.

        Args:
            mu_sigma (torch.Tensor): The output of the encoder

        Returns:
            torch.Tensor: The mu vector
        """

        return mu_sigma[:, :self.latent_dim]

    def _get_sigma(self, mu_sigma: torch.Tensor) -> torch.Tensor:
        """Given the output of the encoder, returns the sigma vector.

        Args:
            mu_sigma (torch.Tensor): The output of the encoder

        Returns:
            torch.Tensor: The sigma vector
        """

        return mu_sigma[:, self.latent_dim:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu_sigma = self.encoder(x)

        # Reparametrization trick
        epsilon = torch.randn_like(self._get_sigma(mu_sigma))
        z_new = self._get_mu(mu_sigma) + self._get_sigma(mu_sigma) * epsilon
        z_rec = self.decoder(z_new)

        return z_rec, mu_sigma

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.lr_patience, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        X, _, w = batch

        X_rec, mu_sigma = self(X)
        mu, sigma = self._get_mu(mu_sigma), self._get_sigma(mu_sigma)

        rec_loss = (self.rec_loss(X_rec, X, w) * self.rec_loss_weight).sum()
        kl_loss = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2) * self.kl_loss_weight

        self.train_rec_losses.append(rec_loss)
        self.train_kl_losses.append(kl_loss)
        return rec_loss + kl_loss

    def on_train_epoch_end(self) -> None:
        rec_loss = torch.stack(self.train_rec_losses)
        kl_loss = torch.stack(self.train_kl_losses)
        self.log('train_rec_loss', rec_loss.mean())
        self.log('train_kl_loss', kl_loss.mean())
        self.train_rec_losses.clear()
        self.train_kl_losses.clear()

    def validation_step(self, batch, batch_idx):
        X, _, w = batch

        X_rec, mu_sigma = self(X)
        mu, sigma = self._get_mu(mu_sigma), self._get_sigma(mu_sigma)

        rec_loss = (self.rec_loss(X_rec, X, w) * self.rec_loss_weight).sum()
        kl_loss = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2) * self.kl_loss_weight
        loss = rec_loss + kl_loss

        self.val_losses.append(loss)
        self.val_x.append(X)
        self.val_x_rec.append(X_rec)
        return loss

    def log_reconstruction(self):
        x = torch.cat(self.val_x).cpu().detach().numpy()
        x_rec = torch.cat(self.val_x_rec).cpu().detach().numpy()

        df_x = pd.DataFrame(x, columns=self.features)
        df_x_rec = pd.DataFrame(x_rec, columns=self.features)

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
            loss = torch.stack(self.val_losses)
            self.log('val_loss', loss.mean())
            # self.log_reconstruction()

        self.is_sanity_check = False
        self.val_x.clear()
        self.val_x_rec.clear()
        self.val_losses.clear()

    def test_step(self, batch, batch_idx):
        X, y, w = batch

        X_rec, mu_sigma = self(X)
        mu, sigma = self._get_mu(mu_sigma), self._get_sigma(mu_sigma)

        rec_loss = (self.rec_loss(X_rec, X, w) * self.rec_loss_weight).sum(dim=1)
        kl_loss = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2) * self.kl_loss_weight
        loss = rec_loss + kl_loss

        self.test_losses.append(loss)
        self.test_labels.append(y)
        return loss

    def _plot_roc_curve(self, labels: torch.Tensor, loss: torch.Tensor) -> None:
        plt.clf()
        fpr, tpr, _ = roc_curve(labels, loss)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("auc = {:.4f}".format(roc_auc_score(labels, loss)))
        plt.savefig(f"{self.save_as}_roc_curve_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

    def _plot_loss_hist(self, loss: torch.Tensor, labels: torch.Tensor) -> None:
        plt.clf()
        plt.hist(loss[labels == BKG], bins=120, density=True, alpha=0.5, label="Background")
        plt.hist(loss[labels == SIG], bins=120, density=True, alpha=0.5, label="Signal")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.yscale("log")
        plt.legend()
        plt.savefig(f"{self.save_as}_loss_hist_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

    def on_test_epoch_end(self):
        loss = torch.cat(self.test_losses).cpu().numpy()
        labels = torch.cat(self.test_labels).cpu().numpy()

        self._plot_roc_curve(labels, loss)
        self._plot_loss_hist(loss, labels)

        self.test_losses.clear()
        self.test_labels.clear()


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

    vae = LitVariationaAutoEncoder("hg3000_hq1000", x_train.shape[1], 4, [64, 32, 16, 12], [fet for fet in train.columns if fet not in ['is_signal', 'weight']],rec_loss="mse", lr_patience=10, rec_loss_weight=1e4)

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

    vae_trainer.test(vae, vae_test_loader)
    vae_trainer.fit(vae, vae_train_loader, vae_val_loader)
    vae_trainer.test(vae, vae_test_loader)
