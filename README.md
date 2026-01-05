# supernote-gan

A GAN (Generative Adversarial Network) trained to generate images in the style of notes from a Ratta Supernote.

## Prerequisites

To use the `conversion.sh` script for dataset creation, you need to have `supernote-tool` installed and accessible in your system's PATH. You can find installation instructions for `supernote-tool` on its official repository or documentation.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/EthanDGee/supernote-gan.git
    cd supernote-gan
    ```
2.  Install dependencies. This project uses poetry for dependency management.
    ```bash
    poetry install
    ```

## Dataset Creation

The quality of the generated images depends on the dataset you provide. To create a dataset from your Supernote files:

1.  Export your Supernote notes and place them in a directory named `Note` in the root of this project.
2.  Run the conversion script:
    ```bash
    ./conversion.sh
    ```
    This script will use `supernote-tool` to convert your `.note` files into `.png` images and place them in the `converted/` directory. This directory will be used for training the GAN.

## Usage

To train the model, run the main script:

```bash
poetry run python src/main.py
```

Checkpoints of the model will be saved in the `checkpoints/` directory, and generated images will be saved in the `images/` directory.

## Configuration

The project supports fine-grained tuning through environment variables loaded from a `.env` file in the project's root directory. A `.env.example` file is provided as a template. You can create a `.env` file by copying and modifying the example:

```bash
cp .env.example .env
```

Here are the configurable parameters and their default values:

*   **`IMAGES_DIR`**: Directory where converted images are stored for training.
    *   Default: `converted`
*   **`NOTE_SIZE`**: The target size for the Supernote images (width, height).
    *   Default: `(351, 468)` or 1/4 of the A5/Nomad
*   **`DEVICE`**: The device to use for training (e.g., `cpu`, `cuda`, `auto`).
    *   Default: `auto`
*   **`NUM_WORKERS`**: Number of worker processes for data loading.
    *   Default: `4`
*   **`BATCH_SIZE`**: Training batch size.
    *   Default: `32`
*   **`NUM_EPOCH`**: Number of training epochs.
    *   Default: `100`
*   **`LEARNING_DISCRIM`**: Learning rate for the discriminator.
    *   Default: `0.001`
*   **`LEARNING_GENER`**: Learning rate for the generator.
    *   Default: `0.001`
*   **`LATENT_SIZE`**: Size of the latent space vector for the generator.
    *   Default: `100`
*   **`SAVE_DIR`**: Directory to save model checkpoints.
    *   Default: `checkpoints`
*   **`SAVE_INTERVAL`**: Interval (in epochs) at which to save model checkpoints.
    *   Default: `5`

## Contributing

We welcome contributions to the `supernote-gan` project! If you'd like to contribute, please feel free to:

*   Report bugs
*   Suggest new features
*   Submit pull requests

## License

This project is licensed under the GNU General Public License v3.0. See the `LICENSE` file for more details.
