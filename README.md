# Facenet Probabilistic Face Embedding (PFE)

This project implements a structured, modular PyTorch codebase for **Probabilistic Face Embeddings (PFE)**. PFE represents face images as Gaussian distributions $\mathcal{N}(\mu, \sigma^2 \mathbf{I})$ rather than fixed points, which allows modeling image uncertainty (e.g., noise, blur, occlusions).

---

## 📂 Project Structure

```
secure_face_recognition/
├── configs/
│   └── config.yaml          # Hyperparameters and dataset paths
├── data/
│   ├── Dataset.csv          # File mapping image IDs to identity labels
│   └── Faces/Faces/         # Directory containing the face JPG images
├── src/
│   ├── __init__.py
│   ├── model.py             # FacenetProbabilisticEmbedding model
│   ├── loss.py              # Mutual Likelihood Score (MLS) loss
│   ├── dataset.py           # RealFacePairDataset image loader
│   ├── train.py             # Training loop for the variance head
│   └── inference.py         # Inference and similarity check demo
├── tests/
│   └── test_model.py        # PyTest suite (clamping, shapes, MLS math)
├── requirements.txt         # Package dependencies
└── README.md                # Command guide
```

---

## 🛠️ Command Guide

### 1. Installation
Install the necessary python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running Unit Tests
Execute the test suite to verify model outputs, shapes, gradients, and loss computations:
```bash
pytest tests/test_model.py
```

### 3. Training the Model
Train the uncertainty estimation variance head using the positive pairs extracted from the real face dataset:
```bash
python -m src.train
```
*This script optimizes only the variance head parameters (keeping the InceptionResnetV1 backbone frozen) and logs the training loss alongside variance estimations on clean and noisy images.*

### 4. Running the Similarity & Match Demo
Run inference on real face images to verify estimated uncertainty values and compute Mutual Likelihood Scores (MLS):
```bash
python -m src.inference
```
*This compares:*
* *Same identity (both clean)*
* *Same identity (one clean, one degraded)*
* *Different identities (both clean)*

---

## 🧠 Loss & Evaluation Mechanism

1. **Mutual Likelihood Score (MLS)**: Computes the likelihood that two probabilistic representations belong to the same person.
2. **Uncertainty Clamping**: Estimates the log variance ($\log \sigma^2$), clamped between $[-10.0, 2.0]$ for numerical stability.
