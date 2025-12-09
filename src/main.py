from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from src.constants import (
    BATCH_SIZE,
    DEVICE,
    IMAGES_DIR,
    LATENT_SIZE,
    LEARNING_DISCRIM,
    LEARNING_GENER,
    NUM_EPOCH,
    SAVE_DIR,
    SAVE_INTERVAL,
)
from src.dataset.dataloader import create_dataloader
from src.models.discriminator import Discriminator
from src.models.generator import Generator


class GANTrainer:
    def __init__(
        self,
        images_dir: str = IMAGES_DIR,
        batch_size: int = BATCH_SIZE,
        latent_size: int = LATENT_SIZE,
        lr_g: float = LEARNING_GENER,
        lr_d: float = LEARNING_DISCRIM,
        beta1: float = 0.5,
        beta2: float = 0.999,
        device: str = DEVICE,
        save_dir: str = SAVE_DIR,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.latent_dim = latent_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Initialize models
        self.generator = Generator(latent_size=latent_size).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=(beta1, beta2)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(beta1, beta2)
        )

        # Loss function
        self.criterion = nn.BCELoss()

        # Data loader
        self.dataloader = create_dataloader(
            batch_size=batch_size, images_dir=images_dir
        )

        # Fixed noise for visualization
        self.fixed_noise = torch.randn(16, latent_size, device=self.device)

        # Training history
        self.g_losses = []
        self.d_losses = []

        print(f"Training on device: {self.device}")
        print(f"Batch size: {batch_size}")

    def train_step(self, real_images):
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)

        # Labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # Train Discriminator
        self.optimizer_d.zero_grad()

        # Real images
        output_real = self.discriminator(real_images)
        loss_real = self.criterion(output_real, real_labels)

        # Fake images
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)
        output_fake = self.discriminator(fake_images.detach())
        loss_fake = self.criterion(output_fake, fake_labels)

        # Total discriminator loss
        loss_d = loss_real + loss_fake
        loss_d.backward()
        self.optimizer_d.step()

        # Train Generator
        self.optimizer_g.zero_grad()

        # Generate fake images
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)
        output = self.discriminator(fake_images)

        # Generator wants discriminator to think fake images are real
        loss_g = self.criterion(output, real_labels)
        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.item(), loss_d.item()

    def save_samples(self, epoch):
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise)
            fake_images = (fake_images + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]

            # Save grid of images
            vutils.save_image(
                fake_images,
                self.save_dir / f"fake_samples_epoch_{epoch:04d}.png",
                nrow=4,
                normalize=True,
            )
        self.generator.train()

    def save_models(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "optimizer_g_state_dict": self.optimizer_g.state_dict(),
                "optimizer_d_state_dict": self.optimizer_d.state_dict(),
                "g_losses": self.g_losses,
                "d_losses": self.d_losses,
            },
            self.save_dir / f"checkpoint_epoch_{epoch:04d}.pth",
        )

    def save_losses(self):
        # Save losses to text file
        with open(self.save_dir / "training_losses.csv", "w") as f:
            f.write("Iteration,Generator_Loss,Discriminator_Loss\n")
            for i, (g_loss, d_loss) in enumerate(zip(self.g_losses, self.d_losses)):
                f.write(f"{i},{g_loss:.6f},{d_loss:.6f}\n")

    def train(self, num_epochs: int = NUM_EPOCH, save_interval: int = SAVE_INTERVAL):
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.generator.train()
            self.discriminator.train()

            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0

            for batch_idx, real_images in enumerate(self.dataloader):
                loss_g, loss_d = self.train_step(real_images)

                epoch_g_loss += loss_g
                epoch_d_loss += loss_d
                num_batches += 1

                # Print progress every 10 batches
                if batch_idx % 10 == 0:
                    print(
                        f"  Batch {batch_idx}: G_Loss: {loss_g:.4f}, D_Loss: {loss_d:.4f}"
                    )

                # Store losses
                self.g_losses.append(loss_g)
                self.d_losses.append(loss_d)

            # Calculate epoch averages
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches

            print(
                f"Epoch {epoch + 1}/{num_epochs} - G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}"
            )

            # Save samples and models
            if (epoch + 1) % save_interval == 0:
                self.save_samples(epoch + 1)
                self.save_models(epoch + 1)
                self.save_losses()

        # Save final models and samples
        self.save_samples(num_epochs)
        self.save_models(num_epochs)
        self.save_losses()

        print("Training completed!")


def main():
    # Training configuration

    # Create trainer
    trainer = GANTrainer()

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
