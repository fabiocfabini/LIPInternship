from typing import Literal, List

import torch
import torch.nn as nn
import pytorch_lightning as pl

from loss_functions import mse_loss, mae_loss


class LitVariationaAutoEncoder(pl.LightningModule):
    def __init__(self, 
                input_dim: int,
                latent_dim: int, 
                arch: List[int], 
                rec_loss: Literal["mse", "mae"] = "mse", 
                rec_loss_weight: float = 1.0, 
                kl_loss_weight: float = 1.0, 
                lr_patience: int = 5
        ):
        super().__init__()

        self.arch = arch
        self.latent_dim = latent_dim
        self.input_dim = input_dim

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

    def _encoder(self) -> nn.Sequential:
        """The encoder recieves [input_dim] features as input 
        and outputs [latent_dim * 2] features. It must 
        have the [arch] architecture.

        Returns:
            nn.Sequential: The encoder
        """

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.arch[0]),
            nn.ReLU()
        )

        for i in range(1, len(self.arch)):
            self.encoder.add_module(f"fc({i})", nn.Linear(self.arch[i - 1], self.arch[i]))
            self.encoder.add_module(f"A({i})", nn.ReLU())

        self.encoder.add_module(f"A({i+1})", nn.ReLU())
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
            nn.ReLU()
        )

        for i in range(len(self.arch) - 1, 0, -1):
            self.decoder.add_module(f"fc({i})", nn.Linear(self.arch[i], self.arch[i - 1]))
            self.decoder.add_module(f"A({i})", nn.ReLU())

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

        rec_loss = self.rec_loss(X_rec, X, w) * self.rec_loss_weight
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

        rec_loss = self.rec_loss(X_rec, X, w) * self.rec_loss_weight
        kl_loss = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu**2 - sigma**2) * self.kl_loss_weight
        loss = rec_loss + kl_loss

        self.val_losses.append(loss)
        return loss

    def on_validation_epoch_end(self) -> None:
        loss = torch.stack(self.val_losses)
        self.log('val_loss', loss.mean())

        self.val_losses.clear()