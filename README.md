[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/yyyuan2004/main-msibruisenet)

# MSI-Bruise-Seg

**Apple Multispectral Image Bruise Segmentation**
基于近红外多光谱图像（MSI）的苹果瘀伤像素级语义分割

> Pixel-level semantic segmentation of apple bruise defects from 9-band NIR multispectral images (565-730 nm).

---

## Quick Start

```bash
pip install -r requirements.txt

# Single run (train + eval + plots)
python train_eval.py --config configs/baseline.yaml --seed 42

# 5-fold cross-validation
python train_eval.py --config configs/baseline.yaml --seed 42 --kfold 5

# Full ablation (17 configs x 3 seeds)
bash run_ablation.sh

# Generate comparison plots (after ablation)
python scripts/plot_ablation.py
```

### Gate band-finding (k = 1, 2, 3)

Run **only** the learnable band gate (no metrics, no baselines) to read off the
physical bands it keeps, repeated 30× per `k`. Early stopping is on gate-selection
stability (anneal `tau`, then stop once the chosen bands stop changing):

```bash
bash run_gate_band_finder.sh --data_dir /root/autodl-tmp/datasets/185_9bands
# or directly:
python scripts/gate_band_finder.py --ks 1,2,3 --runs 30
python scripts/gate_band_finder.py --plots_only   # rebuild the figure from existing runs
```

Writes to `outputs/gate_band_finder/`: per-run `selected_bands.json`, an aggregated
`band_selection_frequency.{json,csv}`, and the per-band frequency figure
`band_frequency.png` (+ `band_frequency_by_k.png`).

### PCA baseline (leak-free, one command)

The PCA 9&rarr;3 projection must be fit on the **training split only** — fitting on
all 185 images leaks val/test spectra. This driver does the whole thing in one
step (fit train-only PCA &rarr; results package &rarr; train + eval, same seed = same split):

```bash
bash run_pca_baseline.sh --data_dir /root/autodl-tmp/datasets/185_9bands --seed 42
# or directly:
python scripts/run_pca_baseline.py --data_dir <root> --seed 42
python scripts/run_pca_baseline.py --data_dir <root> --seed 42 --package_only  # no training
```

Writes to `outputs/pca_baseline_seed<seed>/`: the train-only `pca_matrix.npz`, the
training/eval outputs, and a zipped `pca_package/` with 7 CSVs — PCA loadings &
band contributions, explained variance, per-region (apple/healthy/defect) band
statistics, defect-vs-healthy separability (Cohen's d + pixel AUC), 9&rarr;3
reconstruction error, apple-region band correlation, and a per-image audit
(apple/defect pixel counts + which split each image is in).

### Concrete-Autoencoder baseline (learnable band selection)

`ConcreteSelect` (Balin et al., ICML 2019) is a drop-in sibling of the hard
top-k `DiagonalBandGate`: same `band_gate` slot, same tau schedule, same
`selected_bands.json` logging and deploy export. `configs/concrete_k3.yaml` is
identical to `gate_k3.yaml` except `band_select_type: "concrete"`, so the
comparison isolates the selection mechanism (concrete relaxation vs. hard
top-k + straight-through) with no other confound.

```bash
# Paired multi-seed comparison for the paper table
for s in 42 123 456 2024 7 13 99; do
  python train.py --config configs/gate_k3.yaml     --seed $s --output_dir outputs/gate_k3_seed$s
  python train.py --config configs/concrete_k3.yaml --seed $s --output_dir outputs/concrete_k3_seed$s
done

# Or as an extra arm in the batch runner
bash run_gate_batch.sh --concrete
```

Aggregate with `aggregate_results.py` and run a paired test (Wilcoxon /
paired t) on per-seed class-1 IoU — both arms share seeds, so runs are paired.

---

## Dataset

```
/root/autodl-tmp/datasets/185_9bands/
├── images/    # .npy, shape (H, W, 9), float32
└── masks/     # .npy or .png, shape (H, W), binary
```

185 annotated samples, 9 NIR bands. Change path in `configs/_defaults.yaml`.

---

## Model Zoo

| Category | Config | Architecture | 
|----------|--------|--------------|
| **Custom** | `baseline` | MobileNetV2 + UNet decoder |
| | `spconv_se` | + SpectralConv1D + SE at every skip |
| | `se` | + SE at every skip |
| | `spconv` | + SpectralConv1D after S1 |
| | `input_spconv` | + SpectralConv1D on the raw 9-band input |
| | `pca_baseline` | PCA-3ch + MobileNetV2-UNet |
| **SMP** | `smp_unet_resnet18` | U-Net + ResNet18 |
| | `smp_unet_resnet34` | U-Net + ResNet34 |
| | `smp_unetplusplus_resnet34` | UNet++ + ResNet34 |
| | `smp_linknet_resnet34` | Linknet + ResNet34 |
| | `smp_manet_resnet34` | MAnet + ResNet34 |
| | `smp_deeplabv3plus_mobilenetv2` | DeepLabV3+ + MobileNetV2 |
| | `smp_fpn_efficientnetb0` | FPN + EfficientNet-B0 |
| **Transformer** | `topformer_t/s/b` | TopFormer (1.4 / 3.1 / 5.1M) |
| | `seaformer_t/s/b` | SeaFormer (1.7 / 4 / 8M) |
| **Real-time** | `pidnet_s/m` | PIDNet (7.6 / 23M) |

---

## Training Settings

All configs inherit from [`configs/_defaults.yaml`](configs/_defaults.yaml):

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (wd=1e-4) |
| LR schedule | 5e-4 &rarr; 1e-6 (cosine) |
| Epochs / Patience | 300 / 70 (early stop on class-1 IoU) |
| Batch / Workers | 32 / 8 |
| Loss | 0.5 CE + 0.5 Dice |
| Input | 512&times;512, crop 384&times;384 |
| Augmentation | H-flip, V-flip, {90, 180, 270}&deg; rotation |
| Bands | `[0, 2, 4, 5]` (4 of 9) |

Each experiment config only overrides what differs:

```yaml
# configs/baseline.yaml
_base: _defaults.yaml
experiment_name: "baseline"
model:
  encoder_name: "mobilenetv2"
  skip_module: "none"
  ...
```

---

## `in_channels` Adaptation

| Encoder | Strategy |
|---------|----------|
| Custom (MobileNetV2 / V3 / EfficientNet-B0) | Copy first 3 pretrained channels + Kaiming init for extra |
| SMP models | SMP built-in weight averaging/repeating |
| TopFormer / SeaFormer / PIDNet | Same as Custom |

---

## Project Structure

```
.
├── configs/                 # _defaults.yaml + 19 experiment configs
├── data/
│   ├── dataset.py           # MSIDataset (.npy + band selection)
│   ├── augment.py           # Spatial augmentations
│   └── split.py             # 7:3 split / k-fold
├── model/
│   ├── encoder.py           # MobileNetV2/V3/EfficientNet-B0
│   ├── decoder.py           # UNet decoder (skip: none/se)
│   ├── modules.py           # SEBlock, SpectralConv1D
│   ├── smp_models.py        # SMP wrapper
│   ├── topformer.py         # TopFormer
│   ├── seaformer.py         # SeaFormer
│   ├── pidnet.py            # PIDNet
│   ├── model.py             # build_model() factory
│   └── loss.py              # CE + Dice / Focal
├── utils/
│   ├── config.py            # YAML loader with _base inheritance
│   ├── metrics.py           # IoU / F1 / Precision / Recall
│   └── spectral_analysis.py # Band-level statistics
├── scripts/
│   ├── plot_ablation.py     # Publication-quality comparison plots
│   └── band_*.py            # Band selection search scripts
├── train.py                 # Training loop (resume + early stopping)
├── eval.py                  # Evaluation + TP/FP/FN error analysis
├── train_eval.py            # One-click: train → eval → plots
├── run_ablation.sh          # Full ablation automation
├── run_band_search.sh       # Band search automation
└── aggregate_results.py     # Multi-seed result aggregation
```

---

## Outputs

```
outputs/<config>_seed<seed>/
├── checkpoints/best_model.pth
├── training_log.json
├── visualization/           # loss/IoU/F1 curves
└── eval_results/
    ├── results.json
    ├── confusion_matrix.png
    ├── visualizations/      # prediction vs GT
    └── error_analysis/      # TP(green) / FP(red) / FN(blue) overlay

outputs/ablation_plots/      # after run_ablation.sh
├── iou_boxplot.png
├── f1_boxplot.png
├── grouped_bar.png
└── params_vs_iou.png
```

---

## Resume & Checkpointing

- `last_checkpoint.pth` saved every 10 epochs (model + optimizer + scheduler + RNG state)
- `training_done.flag` skips completed training on re-run
- `done.flag` marks fully completed runs; `run_ablation.sh` auto-skips them
