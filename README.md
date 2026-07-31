# Pneumonia-MultiModel-XAI

A PyTorch-based deep learning framework for multi-class pneumonia classification using chest X-ray images.

## Project Objective

This project compares four state-of-the-art deep learning architectures for multi-class pneumonia classification:

- ResNet50 (Baseline CNN)
- EfficientNet-B2
- ConvNeXt-Tiny
- Swin Transformer-Tiny

All models are trained under identical experimental settings to ensure a fair comparison. Model predictions are interpreted using Grad-CAM for explainable artificial intelligence (XAI).

## Dataset

- Pediatric Chest X-ray Dataset (Kermany et al.)
- Classes:
  - NORMAL
  - BACTERIA
  - VIRUS
- Stratified Train / Validation / Test Split

## Project Structure

```text
dataset/
models/
results/
logs/
notebooks/
```

## Models

- ResNet50
- EfficientNet-B2
- ConvNeXt-Tiny
- Swin Transformer-Tiny

## Explainability

- Grad-CAM

## Framework

- Python
- PyTorch
- Torchvision
- CUDA

## Status

🚧 Journal Version 2 (In Progress)

## License

For research and educational purposes.
