# Pneumonia-MultiModel-XAI

Three-class pediatric pneumonia classification (NORMAL / BACTERIA / VIRUS) from chest X-rays, comparing ResNet-50, ConvNeXt-Tiny, EfficientNet-B2, and PneumoXNet (an EfficientNet-B2 backbone with CBAM, multi-scale fusion, Adaptive Feature Fusion, and residual enhancement), with multi-seed evaluation, ablation, Grad-CAM explainability, and cross-dataset testing.

Code for the paper:

> **PneumoXNet: An Empirical Study of Attention and Feature Fusion on EfficientNet-B2 for Three-Class Pediatric Pneumonia Classification, with Multi-Seed, Cross-Dataset, and Explainability Analysis**
> Jaimul Haque, Ezabul Alam, Akhlakur Rahman Meraj
> Department of Computer Science and Engineering, Bangladesh University of Business and Technology (BUBT)

> **Not for clinical use.** This is research code. It has not been validated by radiologists or in a clinical setting.

## Main findings

The results are reported honestly, including the negative ones:

- PneumoXNet did **not** beat its own backbone in-distribution. Its mean test accuracy over three seeds was the lowest of the four models. Only the gap to EfficientNet-B2 was statistically significant (paired t-test, p = 0.017, n = 3).
- In the (single-run, validation-set) ablation, removing CBAM or AFF did not lower accuracy; only removing residual enhancement did, by 0.34 points (about 3 images), which is too small to be conclusive.
- On an external dataset, PneumoXNet's accuracy was close to EfficientNet-B2's, and ConvNeXt-Tiny was best.

## Models compared

| Model | Notes |
|---|---|
| ResNet-50 | Baseline CNN, ImageNet-pretrained |
| ConvNeXt-Tiny | ImageNet-pretrained |
| EfficientNet-B2 | ImageNet-pretrained, plain backbone baseline |
| PneumoXNet | EfficientNet-B2 + CBAM + multi-scale fusion + AFF + residual enhancement |

All four models are trained and evaluated under the same protocol: 260x260 input, same split, same training configuration. See the paper (Section III and Table II) for the full configuration.

## Results

Test accuracy over three training seeds (42, 123, 2024), mean ± SD:

| Model | Accuracy |
|---|---|
| EfficientNet-B2 | 85.29% ± 0.80% |
| ConvNeXt-Tiny | 84.79% ± 0.35% |
| ResNet-50 | 83.85% ± 1.37% |
| PneumoXNet | 83.47% ± 1.21% |

Complexity (computed from the model definitions; latency is single-thread CPU, batch size 1):

| Model | Params | FLOPs | CPU latency |
|---|---|---|---|
| EfficientNet-B2 | 7.71M | 2.11G | 50.6 ms |
| ResNet-50 | 23.51M | 12.03G | 120.6 ms |
| ConvNeXt-Tiny | 27.82M | 11.72G | 109.1 ms |
| PneumoXNet | 36.81M | 6.67G | 130.7 ms |

External test (COVID-19 Radiography Database, collapsed to NORMAL vs. PNEUMONIA, 17,549 images, single trained model per architecture):

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| ConvNeXt-Tiny | 89.02% | 0.8748 | 0.9419 |
| PneumoXNet | 85.54% | 0.8403 | 0.9211 |
| EfficientNet-B2 | 85.17% | 0.8389 | 0.9224 |
| ResNet-50 | 82.87% | 0.8222 | 0.9062 |

## Dataset

- **Training / in-distribution evaluation:** Pediatric Chest X-ray dataset (Kermany et al.), 5,856 images, classes NORMAL, BACTERIA, VIRUS. Stratified 70/15/15 split: 4,099 train / 878 validation / 879 test.
- **External evaluation:** COVID-19 Radiography Database.

The datasets are not redistributed here. Download them from their original public sources and place them under `dataset/`.

## Repository structure

```
dataset/processed_dataset/   processed data used for training and evaluation
notebooks/                   training, evaluation, ablation, Grad-CAM, cross-dataset notebooks
results/                     saved metrics and result tables
evaluation_outputs/          evaluation outputs (predictions, confusion matrices, etc.)
figures/                     figures used in the paper
```

## Setup

Environment used in the paper: Python 3.12.3, PyTorch 2.13.0 (+cu126), torchvision, on a single NVIDIA RTX 3050 (8 GB).

```bash
git clone https://github.com/jaimulhaque/Pneumonia-MultiModel-XAI.git
cd Pneumonia-MultiModel-XAI
pip install torch torchvision numpy pandas scikit-learn matplotlib
```

## Usage

1. Download the Kermany dataset and put it under `dataset/`.
2. Open the notebooks in `notebooks/` and run them in numeric order.
3. For the external test, also download the COVID-19 Radiography Database.

## Known limitations

See the Limitations section of the paper. In short: a single in-distribution dataset, three seeds, a single-run ablation, Grad-CAM not reviewed by a radiologist, and an external dataset that differs in population and labeling from the training data.

## Status

- **Version 1:** the code and results behind the paper above.
- **Version 2 (in progress):** patient-level data splits, additional backbones, and more seeds.

## License

MIT. See [LICENSE](LICENSE).
