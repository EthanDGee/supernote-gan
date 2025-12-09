import torch

from src.models.discriminator import Discriminator


def test_discriminator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = Discriminator().to(device)

    # Test with random images
    batch_size = 4
    images = torch.randn(batch_size, 3, 351, 468).to(device)

    with torch.no_grad():
        output = discriminator(images)

    print(f"Discriminator output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 1)")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    assert output.shape == (batch_size, 1), f"Unexpected output shape: {output.shape}"
    assert 0 <= output.min() and output.max() <= 1, "Output should be in [0, 1] range"
    print("Discriminator test passed!")


if __name__ == "__main__":
    test_discriminator()
