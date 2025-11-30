# Animal Classification Model

A Streamlit-based web application for classifying animal images using a pre-trained TensorFlow Keras model.

## Features

- Image classification for 15 animal types:
  - Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra
- Real-time predictions with accuracy scores
- User-friendly Streamlit interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/prajakta-ronit/Animal-Classification.git
cd Animal-Classification
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # On Windows PowerShell
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then enter an image path or filename to classify the animal.

## Model

The project uses a pre-trained Keras model (`Image_classifier.keras`) trained on a dataset of animal images.

## Dataset

Training data is organized in `dataset1/` with `train/`, `val/`, and `test/` subdirectories.

## Requirements

- Python 3.8+
- TensorFlow
- Streamlit
- NumPy

See `requirements.txt` for specific versions.
