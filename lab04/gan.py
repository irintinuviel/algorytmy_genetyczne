"""
DCGAN dla obrazow MRI 64x64 w skali szarosci (BRISC2025).

Architektura wg DCGAN (Radford i in.): generator z ConvTranspose2d + Tanh,
dyskryminator z Conv2d + LeakyReLU. Trenujemy bezwarunkowo na wszystkich
obrazach treningowych. Wytrenowany generator jest pozniej zamrazany i
przeszukiwany przez algorytm genetyczny w przestrzeni latentnej.
"""

import os
import torch
import torch.nn as nn
import torchvision.utils as vutils

from data import gan_loader

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
G_WEIGHTS_PATH = os.path.join(RESULTS_DIR, "generator.pt")
D_WEIGHTS_PATH = os.path.join(RESULTS_DIR, "discriminator.pt")

LATENT_DIM = 100   # wymiar wektora z (chromosom dla GA)
NGF = 64           # bazowa liczba filtrow generatora
NDF = 64           # bazowa liczba filtrow dyskryminatora


class Generator(nn.Module):
    """z (LATENT_DIM) -> obraz 1x64x64 w zakresie [-1, 1]."""

    def __init__(self, latent_dim=LATENT_DIM, ngf=NGF):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # z: (latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),               # 4x4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),               # 8x8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),               # 16x16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),                   # 32x32
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh(),                                            # 64x64
        )

    def forward(self, z):
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    """Obraz 1x64x64 -> skalar (logit prawdziwosci)."""

    def __init__(self, ndf=NDF):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),                      # 32x32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),  # 16x16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),  # 8x8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),  # 4x4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),           # 1x1
        )

    def forward(self, x):
        return self.net(x).view(-1)


def _weights_init(m):
    """Inicjalizacja wag wg zalecen DCGAN."""
    cn = m.__class__.__name__
    if "Conv" in cn:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cn:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_gan(epochs=30, lr=2e-4, beta1=0.5, batch_size=128,
              device="cpu", seed=42, sample_every=5, only_class=None):
    torch.manual_seed(seed)
    loader = gan_loader(batch_size=batch_size, only_class=only_class)
    print(f"[gan] obrazow treningowych: {len(loader.dataset)}")

    netG = Generator().to(device)
    netD = Discriminator().to(device)
    netG.apply(_weights_init)
    netD.apply(_weights_init)

    crit = nn.BCEWithLogitsLoss()
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)
    real_label, fake_label = 1.0, 0.0

    for epoch in range(1, epochs + 1):
        for i, real in enumerate(loader):
            real = real.to(device)
            bs = real.size(0)

            # Osobne tensory etykiet -- BCEWithLogitsLoss potrzebuje wartosci
            # celu podczas backward(), wiec nie wolno ich modyfikowac w miejscu.
            real_lbl = torch.full((bs,), real_label, device=device)
            fake_lbl = torch.full((bs,), fake_label, device=device)

            # --- D: max log(D(x)) + log(1 - D(G(z))) ---
            netD.zero_grad()
            errD_real = crit(netD(real), real_lbl)
            noise = torch.randn(bs, LATENT_DIM, 1, 1, device=device)
            fake = netG(noise)
            errD_fake = crit(netD(fake.detach()), fake_lbl)
            errD = errD_real + errD_fake
            errD.backward()
            optD.step()

            # --- G: max log(D(G(z))) ---
            netG.zero_grad()
            errG = crit(netD(fake), real_lbl)
            errG.backward()
            optG.step()

        print(f"[gan] epoka {epoch}/{epochs}  loss_D={errD.item():.4f}  loss_G={errG.item():.4f}")

        if epoch % sample_every == 0 or epoch == epochs:
            _save_samples(netG, fixed_noise, epoch)

    torch.save(netG.state_dict(), G_WEIGHTS_PATH)
    torch.save(netD.state_dict(), D_WEIGHTS_PATH)
    print(f"[gan] zapisano generator -> {G_WEIGHTS_PATH}")
    print(f"[gan] zapisano dyskryminator -> {D_WEIGHTS_PATH}")
    return netG


@torch.no_grad()
def _save_samples(netG, noise, epoch):
    netG.eval()
    fake = netG(noise).detach().cpu()
    path = os.path.join(RESULTS_DIR, f"gan_samples_epoch{epoch:03d}.png")
    vutils.save_image(fake, path, normalize=True, nrow=8)
    netG.train()
    print(f"[gan]   probki -> {path}")


def load_generator(device="cpu"):
    """Wczytuje zamrozony generator do uzytku przez GA."""
    netG = Generator().to(device)
    netG.load_state_dict(torch.load(G_WEIGHTS_PATH, map_location=device))
    netG.eval()
    for p in netG.parameters():
        p.requires_grad_(False)
    return netG


def load_discriminator(device="cpu"):
    """Wczytuje zamrozony dyskryminator (czlon realizmu w ocenie GA).

    Zwraca None, jesli wagi nie istnieja (np. GAN trenowany przed dodaniem
    zapisu D) -- GA dziala wtedy bez czlonu realizmu opartego o D.
    """
    if not os.path.exists(D_WEIGHTS_PATH):
        return None
    netD = Discriminator().to(device)
    netD.load_state_dict(torch.load(D_WEIGHTS_PATH, map_location=device))
    netD.eval()
    for p in netD.parameters():
        p.requires_grad_(False)
    return netD


if __name__ == "__main__":
    train_gan()
