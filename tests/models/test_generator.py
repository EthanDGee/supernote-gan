import torch

from src.models.generator import Generator


def test_generator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)

    # Test with random noise
    batch_size = 4
    z = torch.randn(batch_size, 100).to(device)

    with torch.no_grad():
        output = generator(z)

    print(f"Generator output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 3, 351, 468)")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    assert output.shape == (batch_size, 3, 351, 468), (
        f"Unexpected output shape: {output.shape}"
    )
    print("Generator test passed!")
