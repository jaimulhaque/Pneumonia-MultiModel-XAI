# Pneumonia-MultiModel-XAI

Three-class pediatric pneumonia classification (NORMAL / BACTERIA / VIRUS) from chest X-rays, comparing ResNet-50, ConvNeXt-Tiny, EfficientNet-B2, and PneumoXNet (an EfficientNet-B2 backbone with CBAM, multi-scale fusion, Adaptive Feature Fusion, and residual enhancement), with multi-seed evaluation, ablation, Grad-CAM explainability, and cross-dataset testing.

Code for the paper:

> **PneumoXNet: An Empirical Study of Attention and Feature Fusion on EfficientNet-B2 for Three-Class Pediatric Pneumonia Classification, with Multi-Seed, Cross-Dataset, and Explainability Analysis**
> Jaimul Haque, Ezabul Alam, Akhlakur Rahman Meraj
> Department of Computer Science and Engineering, Bangladesh University of Business and Technology (BUBT)

> **Not for clinical use.** This is research code. It has not been validated by radiologists or in a clinical setting.

## Authors

- **Jaimul Haque** ([@jaimulhaque](https://github.com/jaimulhaque))
- **Ezabul Alam**
- **Akhlakur Rahman Meraj**

Department of Computer Science and Engineering, Bangladesh University of Business and Technology (BUBT), Dhaka, Bangladesh.

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

All four models use the same 260x260 input, the same data split, the same loss (class-weighted cross-entropy with label smoothing 0.1), optimizer type (AdamW), batch size, 20-epoch budget, scheduler, and early stopping. They are **not** identical in every setting: PneumoXNet used stronger augmentation, higher classifier dropout, and separate learning rate/weight decay for its backbone and new modules, while the baselines were not tuned individually (paper Table II and Limitations).

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
dataset/processed_dataset/   processed Kermany data (train / validation / test)
notebooks/                   01-19, see below
results/                     saved metrics and result tables
evaluation_outputs/          evaluation outputs
figures/                     figures used in the paper
```

### Notebooks

| Notebook | Purpose |
|---|---|
| 01 | Environment and data verification |
| 02 | Dataset preparation and stratified 70/15/15 split |
| 11 | Ablation study (paper Table VIII) |
| 12 | ResNet-50, ConvNeXt-Tiny, EfficientNet-B2 at 260px, 3 seeds |
| 13 | External dataset preparation (COVID-19 Radiography Database, NORMAL vs. PNEUMONIA) |
| 14 | PneumoXNet training (run once per seed) |
| 15 | PneumoXNet Grad-CAM explainability |
| 17 | Quantitative XAI metrics (Deletion / Insertion) |
| 18 | Cross-dataset generalization |
| 19 | Confusion matrices, ROC curves, and per-class metrics for the seed-42 checkpoints |
| 03-10, 16 | Earlier development experiments (other image sizes and earlier model versions), kept for history; **not** used for the tables in the paper |

Run 01, 02, then 12 and 14 (for seeds 42, 123, 2024), then the rest.

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

## Citation

The paper is a manuscript in preparation. Until it is published, please cite the repository:

```bibtex
@misc{haque2026pneumoxnet,
  author       = {Haque, Jaimul and Alam, Ezabul and Meraj, Akhlakur Rahman},
  title        = {Pneumonia-MultiModel-XAI: Code for an Empirical Study of Attention and Feature Fusion on EfficientNet-B2 for Three-Class Pediatric Pneumonia Classification},
  year         = {2026},
  howpublished = {\url{https://github.com/jaimulhaque/Pneumonia-MultiModel-XAI}}
}
```

## License

MIT. See [LICENSE](LICENSE).
