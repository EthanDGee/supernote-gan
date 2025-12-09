from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
from PIL import Image

from src.dataset.constants import IMAGES_DIR


class SupernoteImageDataset(Dataset):
    def __init__(self, images_dir: str = IMAGES_DIR):
        self.images_dir = Path(images_dir)
        self.image_size = (351, 468)  # 1/4 of the supernote nomad

        self.transforms = transforms.Compose(
            [transforms.Resize(self.image_size), transforms.ToTensor()]
        )

        self.image_paths = [
            str(p) for p in self.images_dir.glob("*.png") if p.is_file()
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        # TODO: Explore the benifits of black and white

        return self.transforms(image)


def create_dataloader(
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    images_dir: str = IMAGES_DIR,
) -> DataLoader:
    dataset = SupernoteImageDataset(images_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    dataloader = create_dataloader(batch_size=4, shuffle=True)
    # print(f"Found {len(dataloader.dataset)} images")

    batch = next(iter(dataloader))
    print(f"Batch shape: {batch.shape}")
    print(f"Batch dtype: {batch.dtype}")
    print(f"Batch range: [{batch.min():.3f}, {batch.max():.3f}]")
