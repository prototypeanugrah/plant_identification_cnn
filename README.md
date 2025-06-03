# Indian Plant Leaf Species Classification

## Project Overview
This project focuses on the classification of Indian plant leaf species using deep learning. Leveraging a high-quality, open-source dataset, the goal is to build and evaluate image classification models that can accurately identify plant species from leaf images. The project is ideal for research and educational purposes in computer vision, plant biology, and machine learning.

## Dataset: Indian Plant Leaves Species
- **Source:** [Hugging Face Datasets](https://huggingface.co/datasets/avaishnav/Indian-plant-leaves-species)
- **Description:**
  - 542 high-resolution images of leaves from 12 different Indian plant species
  - Suitable for image classification and related computer vision tasks
  - Provided in `imagefolder` format
  - Licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Dataset Card & Citation:**
  - See the [dataset card on Hugging Face](https://huggingface.co/datasets/avaishnav/Indian-plant-leaves-species) for detailed information, usage examples, and citation instructions.
  - **Citation:**
    ```bibtex
    @dataset{vaishnav2024indian,
      author       = {Vaishnav, Anugrah},
      title        = {Indian Plant Leaves Species},
      year         = 2024,
      url          = {https://huggingface.co/datasets/avaishnav/Indian-plant-leaves-species},
      publisher    = {Hugging Face},
      license      = {Apache-2.0}
    }
    ```

## Code Implementation Summary (`hf_training.ipynb`)
The main workflow is implemented in the Jupyter notebook `hf_training.ipynb`, which covers:
- **Data Loading:** Loading the dataset directly from Hugging Face using the `datasets` library.
- **Preprocessing:** Image transformations and preparation for model input.
- **Visualization:** Displaying sample images and class distributions.
- **Model Setup:** Using Hugging Face Transformers to configure a Vision Transformer (ViT) model for image classification.
- **Training:** Fine-tuning the model on the leaf dataset with PyTorch and Hugging Face Trainer.
- **Evaluation:** Assessing model performance and visualizing results.

## Installation & Setup
1. **Install [uv](https://github.com/astral-sh/uv):**
   ```bash
   pip install uv
   ```
2. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

## Usage
- Run the notebook `hf_training.ipynb` to reproduce the experiments, train the model, and evaluate results.
- You can modify the notebook to experiment with different models or hyperparameters.

## License
This project and dataset are licensed under the Apache-2.0 License. See the [dataset card](https://huggingface.co/datasets/avaishnav/Indian-plant-leaves-species) and repository files for details.
