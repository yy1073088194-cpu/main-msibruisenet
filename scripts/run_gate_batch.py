"""Batch gate-vs-full comparison across many MSI datasets.

For every MSI dataset under a parent folder (each a subdir with images/ + masks/),
this driver:
  1. auto-detects the channel count C from the .npy files and sets num_channels;
  2. trains, for each (seed, k):
       - gate    : M_k = Backbone ∘ G_k (learned prior-free hard top-k selection)
       - random-k: same architecture, gate frozen on a random k-subset (control;
                   toggle with --random_k)
     and once per seed:
       - full    : the full-input baseline (gate off, all C channels = M_{k=B});
  3. writes a gate-vs-full comparison chart and a selected-bands visualization
     per dataset, plus a cross-dataset summary (does the gate help more when the
     candidate spectral range is wider?).

Robustness:
  - Each training is an isolated subprocess; a crash in one run is logged and the
    batch continues. Per-run resume is handled by train.py (last_checkpoint.pth +
    training_done.flag); a finished run is skipped on re-invocation. Progress is
    tracked in batch_manifest.json (atomic writes) and batch.log.

Usage:
    python scripts/run_gate_batch.py \
        --data_root /root/autodl-fs/15 \
        --seeds 42,123 --ks 2,3,4 --random_k

    python scripts/run_gate_batch.py --data_root /root/autodl-fs/15 --dry_run
    python scripts/run_gate_batch.py --data_root /root/autodl-fs/15 --plots_only
"""

import argparse
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Config loading (import utils/config.py by path so we don't pull in the heavy
# package __init__ / torch into the orchestrator process).
# --------------------------------------------------------------------------- #
def load_config(path):
    spec = importlib.util.spec_from_file_location(
        "_gate_cfgloader", os.path.join(PROJECT_ROOT, "utils", "config.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_config(path)


# --------------------------------------------------------------------------- #
# Dataset discovery + channel detection
# --------------------------------------------------------------------------- #
RANGE_RE = re.compile(r"(\d+)-(\d+)")


def parse_range(name):
    """Parse the trailing band range 'AAA-BBB' from a dataset folder name."""
    matches = RANGE_RE.findall(name)
    if matches:
        a, b = matches[-1]
        return int(a), int(b)
    return None


def detect_channels(ds_path, image_dir):
    """Read one .npy (mmap) and return (num_channels, num_samples)."""
    img_root = os.path.join(ds_path, image_dir)
    files = sorted(f for f in os.listdir(img_root) if f.endswith(".npy"))
    if not files:
        raise FileNotFoundError(f"No .npy images in {img_root}")
    arr = np.load(os.path.join(img_root, files[0]), mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"Expected (H,W,C) image, got shape {arr.shape} in {img_root}")
    return int(arr.shape[-1]), len(files)


def discover_datasets(data_root, image_dir, mask_dir, name_filter=None):
    datasets = []
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if not os.path.isdir(path):
            continue
        if not (os.path.isdir(os.path.join(path, image_dir))
                and os.path.isdir(os.path.join(path, mask_dir))):
            continue
        if name_filter and name not in name_filter:
            continue
        C, n = detect_channels(path, image_dir)
        rng = parse_range(name)
        datasets.append({
            "name": name, "path": path, "C": C, "n_samples": n,
            "range": rng, "width": (rng[1] - rng[0]) if rng else C,
        })
    return datasets


def band_labels(C, rng):
    """Approximate HSI band index for each of the C uniformly-sampled channels."""
    if rng:
        return np.linspace(rng[0], rng[1], C).round().astype(int).tolist()
    return list(range(C))


# --------------------------------------------------------------------------- #
# Run bookkeeping
# --------------------------------------------------------------------------- #
def run_tag(mode, k):
    return mode if mode == "full" else f"{mode}_k{k}"


def run_id(dataset, mode, k, seed):
    return f"{dataset}/{run_tag(mode, k)}_seed{seed}"


def out_dir_for(out_root, dataset, mode, k, seed):
    return os.path.join(out_root, dataset, f"{run_tag(mode, k)}_seed{seed}")


def is_done(out_dir):
    return os.path.exists(os.path.join(out_dir, "checkpoints", "training_done.flag"))


def read_metrics(out_dir):
    """Best val IoU(class1) + F1 at that epoch + gate selection, from a run dir."""
    res = {"iou": None, "f1": None, "epoch": None,
           "selected_local": None, "selected_global": None}
    log_path = os.path.join(out_dir, "training_log.json")
    if os.path.exists(log_path):
        try:
            logs = json.load(open(log_path))
            if logs:
                best = max(logs, key=lambda e: e.get("IoU_class1", 0.0))
                res["iou"] = best.get("IoU_class1")
                res["f1"] = best.get("F1_macro")
                res["epoch"] = best.get("epoch")
        except (ValueError, OSError):
            pass
    sb_path = os.path.join(out_dir, "selected_bands.json")
    if os.path.exists(sb_path):
        try:
            d = json.load(open(sb_path))
            res["selected_local"] = d.get("selected_bands_local")
            res["selected_global"] = d.get("selected_bands_global")
        except (ValueError, OSError):
            pass
    return res


def make_run_config(base_cfg, ds, mode, k, seed, epochs, image_dir, mask_dir):
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["data_dir"] = ds["path"]
    cfg["data"]["image_dir"] = image_dir
    cfg["data"]["mask_dir"] = mask_dir
    cfg["data"]["num_channels"] = ds["C"]
    cfg["data"]["band_indices"] = None

    m = cfg["model"]
    if mode == "full":
        m["use_band_gate"] = False
        m["band_gate_k"] = ds["C"]
        m["band_gate_random_select"] = False
    else:
        m["use_band_gate"] = True
        m["band_gate_k"] = k
        m["band_gate_random_select"] = (mode == "random")
        m["band_select_type"] = "concrete" if mode == "concrete" else "topk"
        if mode == "concrete":
            # Paper-faithful exponential schedule endpoints (T0 -> TB), not the
            # gate's cosine 1.0 -> 0.05 inherited from the base config.
            m["band_gate_tau_start"] = 10.0
            m["band_gate_tau_end"] = 0.01

    cfg["experiment_name"] = f"{ds['name']}__{run_tag(mode, k)}"
    if epochs:
        cfg["train"]["num_epochs"] = epochs
    return cfg


# --------------------------------------------------------------------------- #
# Manifest + logging
# --------------------------------------------------------------------------- #
def save_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def append_log(log_path, msg):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------- #
# Training subprocess
# --------------------------------------------------------------------------- #
def run_training(spec, base_cfg, out_root, epochs, image_dir, mask_dir,
                 python_exe, device, log_path):
    ds, mode, k, seed = spec["ds"], spec["mode"], spec["k"], spec["seed"]
    out_dir = out_dir_for(out_root, ds["name"], mode, k, seed)
    os.makedirs(out_dir, exist_ok=True)

    if is_done(out_dir):
        append_log(log_path, f"[skip] {run_id(ds['name'], mode, k, seed)} already complete")
        return "done", read_metrics(out_dir)

    cfg = make_run_config(base_cfg, ds, mode, k, seed, epochs, image_dir, mask_dir)
    cfg_path = os.path.join(out_dir, "run_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)

    cmd = [python_exe, os.path.join(PROJECT_ROOT, "train.py"),
           "--config", cfg_path, "--seed", str(seed), "--output_dir", out_dir]
    if device:
        cmd += ["--device", device]

    append_log(log_path, f"[run ] {run_id(ds['name'], mode, k, seed)} "
                         f"(C={ds['C']}, epochs={cfg['train']['num_epochs']})")
    t0 = time.time()
    run_log = os.path.join(out_dir, "run.log")
    try:
        with open(run_log, "a") as lf:
            lf.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            lf.flush()
            ret = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=lf,
                                 stderr=subprocess.STDOUT).returncode
    except Exception as e:  # noqa: BLE001 - keep the batch alive on any failure
        append_log(log_path, f"[FAIL] {run_id(ds['name'], mode, k, seed)} exception: {e}")
        return "failed", read_metrics(out_dir)

    elapsed = time.time() - t0
    if ret == 0 and is_done(out_dir):
        metrics = read_metrics(out_dir)
        append_log(log_path, f"[done] {run_id(ds['name'], mode, k, seed)} "
                             f"IoU={metrics['iou']} F1={metrics['f1']} "
                             f"sel={metrics['selected_global']} ({elapsed:.0f}s)")
        return "done", metrics
    append_log(log_path, f"[FAIL] {run_id(ds['name'], mode, k, seed)} "
                         f"returncode={ret}, done_flag={is_done(out_dir)} "
                         f"(will retry on next run; see {run_log})")
    return "failed", read_metrics(out_dir)


# --------------------------------------------------------------------------- #
# Plotting (matplotlib optional)
# --------------------------------------------------------------------------- #
def _get_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("WARNING: matplotlib unavailable, skipping plots.")
        return None


def _collect(manifest, dataset, mode, k, seeds):
    vals = {"iou": [], "f1": []}
    for s in seeds:
        rec = manifest["runs"].get(run_id(dataset, mode, k, s))
        if rec and rec.get("status") == "done" and rec.get("iou") is not None:
            vals["iou"].append(rec["iou"])
            if rec.get("f1") is not None:
                vals["f1"].append(rec["f1"])
    return vals


def _mean_std(xs):
    if not xs:
        return float("nan"), 0.0
    return float(np.mean(xs)), float(np.std(xs))


def plot_dataset_comparison(ds, manifest, out_root, ks, seeds, random_k):
    plt = _get_mpl()
    if plt is None:
        return
    name = ds["name"]
    full = _collect(manifest, name, "full", ds["C"], seeds)
    if not full["iou"] and not any(_collect(manifest, name, "gate", k, seeds)["iou"] for k in ks):
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, label in [(axes[0], "iou", "Class-1 IoU"),
                              (axes[1], "f1", "F1 (macro)")]:
        # full baseline as a horizontal reference band
        fm, fs = _mean_std(full[metric])
        if not np.isnan(fm):
            ax.axhline(fm, color="#444", ls="--", lw=1.5,
                       label=f"full input (C={ds['C']}): {fm:.4f}")
            ax.fill_between([min(ks) - 0.5, max(ks) + 0.5], fm - fs, fm + fs,
                            color="#444", alpha=0.12)
        # gate (and concrete / random baselines) vs k; modes with no runs in
        # the manifest are skipped by the all-NaN check below
        for mode, color, marker in [("gate", "#1f77b4", "o"),
                                    ("concrete", "#2ca02c", "^"),
                                    ("random", "#d62728", "s")]:
            if mode == "random" and not random_k:
                continue
            ms, ss = [], []
            for k in ks:
                m, s = _mean_std(_collect(manifest, name, mode, k, seeds)[metric])
                ms.append(m); ss.append(s)
            if not all(np.isnan(ms)):
                ax.errorbar(ks, ms, yerr=ss, marker=marker, color=color,
                            capsize=3, lw=1.8, label=f"{mode}-selected k")
        ax.set_xlabel("k (number of selected bands)")
        ax.set_ylabel(label)
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    rng = f"{ds['range'][0]}-{ds['range'][1]}" if ds["range"] else "?"
    fig.suptitle(f"Gate vs full input — {name}  (range {rng}, C={ds['C']})",
                 fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.join(out_root, name), exist_ok=True)
    path = os.path.join(out_root, name, "comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_dataset_selected_bands(ds, manifest, out_root, ks, seeds):
    plt = _get_mpl()
    if plt is None:
        return
    name, C = ds["name"], ds["C"]
    freq = np.full((len(ks), C), np.nan)
    any_data = False
    for ki, k in enumerate(ks):
        counts = np.zeros(C)
        n = 0
        for s in seeds:
            rec = manifest["runs"].get(run_id(name, "gate", k, s))
            if rec and rec.get("status") == "done" and rec.get("selected_local"):
                for b in rec["selected_local"]:
                    if 0 <= b < C:
                        counts[b] += 1
                n += 1
        if n > 0:
            freq[ki] = counts / n
            any_data = True
    if not any_data:
        return

    labels = band_labels(C, ds["range"])
    fig, ax = plt.subplots(figsize=(max(6, C * 0.5), 1.2 + 0.6 * len(ks)))
    im = ax.imshow(np.nan_to_num(freq), cmap="YlOrRd", vmin=0, vmax=1,
                   aspect="auto")
    ax.set_xticks(range(C))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([f"k={k}" for k in ks])
    ax.set_xlabel("Approx. HSI band index")
    ax.set_title(f"Gate band-selection frequency across seeds — {name}")
    for ki in range(len(ks)):
        for b in range(C):
            if not np.isnan(freq[ki, b]) and freq[ki, b] > 0:
                ax.text(b, ki, f"{freq[ki, b]:.0%}", ha="center", va="center",
                        fontsize=7, color="black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="selection frequency")
    fig.tight_layout()
    os.makedirs(os.path.join(out_root, name), exist_ok=True)
    path = os.path.join(out_root, name, "selected_bands.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_cross_dataset(datasets, manifest, out_root, ks, seeds, summary_k):
    plt = _get_mpl()
    summary = {"summary_k": summary_k, "datasets": []}
    rows = []
    for ds in datasets:
        name = ds["name"]
        fm, _ = _mean_std(_collect(manifest, name, "full", ds["C"], seeds)["iou"])
        gm, _ = _mean_std(_collect(manifest, name, "gate", summary_k, seeds)["iou"])
        gap = (gm - fm) if (not np.isnan(fm) and not np.isnan(gm)) else float("nan")
        rows.append({"name": name, "width": ds["width"], "C": ds["C"],
                     "full_iou": fm, "gate_iou": gm, "gap": gap})
        summary["datasets"].append(rows[-1])
    save_json_atomic(os.path.join(out_root, "master_summary.json"), summary)

    rows = [r for r in rows if not np.isnan(r["gap"])]
    if plt is None or not rows:
        return
    rows.sort(key=lambda r: r["width"])
    names = [f"{r['name']}\n(w={r['width']})" for r in rows]
    gaps = [r["gap"] for r in rows]
    colors = ["#2ca02c" if g >= 0 else "#d62728" for g in gaps]
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * 1.3), 5))
    bars = ax.bar(range(len(rows)), gaps, color=colors, edgecolor="gray")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel(f"IoU(gate, k={summary_k}) − IoU(full)")
    ax.set_title("Gate contribution vs candidate-range width "
                 "(does selection help more on wider ranges?)")
    for bar, g in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.002 if g >= 0 else -0.004),
                f"{g:+.4f}", ha="center",
                va="bottom" if g >= 0 else "top", fontsize=8)
    ymax, ymin = max(gaps + [0.0]), min(gaps + [0.0])
    pad = 0.18 * max(ymax - ymin, 1e-3)
    ax.set_ylim(ymin - pad, ymax + pad)
    fig.tight_layout()
    path = os.path.join(out_root, "cross_dataset_summary.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_specs(datasets, seeds, ks, random_k, concrete):
    specs = []
    for ds in datasets:
        valid_ks = [k for k in ks if 1 <= k < ds["C"]]
        if len(valid_ks) < len(ks):
            skipped = [k for k in ks if k not in valid_ks]
            print(f"  [{ds['name']}] C={ds['C']}: skipping k={skipped} (need 1<=k<C)")
        for seed in seeds:
            specs.append({"ds": ds, "mode": "full", "k": ds["C"], "seed": seed})
            for k in valid_ks:
                specs.append({"ds": ds, "mode": "gate", "k": k, "seed": seed})
                if concrete:
                    specs.append({"ds": ds, "mode": "concrete", "k": k, "seed": seed})
                if random_k:
                    specs.append({"ds": ds, "mode": "random", "k": k, "seed": seed})
    return specs


def main():
    parser = argparse.ArgumentParser(description="Batch gate-vs-full MSI comparison")
    parser.add_argument("--data_root", type=str, default="/root/autodl-fs/15")
    parser.add_argument("--out_root", type=str, default="outputs/gate_batch")
    parser.add_argument("--base_config", type=str, default="configs/gate.yaml")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--ks", type=str, default="3")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override num_epochs (gate and full share the same budget)")
    parser.add_argument("--random_k", action="store_true",
                        help="Also train a frozen random-k control per (k, seed)")
    parser.add_argument("--concrete", action="store_true",
                        help="Also train the Concrete-Autoencoder selection "
                             "baseline (band_select_type=concrete) per (k, seed)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma list of subfolder names to restrict to")
    parser.add_argument("--image_dir", type=str, default="images")
    parser.add_argument("--mask_dir", type=str, default="masks")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--summary_k", type=int, default=None,
                        help="k used for the cross-dataset gap chart (default: min ks)")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--plots_only", action="store_true",
                        help="Skip training; rebuild charts from the manifest")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    ks = sorted(int(x) for x in args.ks.split(",") if x.strip())
    name_filter = set(args.datasets.split(",")) if args.datasets else None
    summary_k = args.summary_k or min(ks)

    base_cfg = load_config(args.base_config)
    os.makedirs(args.out_root, exist_ok=True)
    log_path = os.path.join(args.out_root, "batch.log")
    manifest_path = os.path.join(args.out_root, "batch_manifest.json")
    manifest = {"runs": {}}
    if os.path.exists(manifest_path):
        try:
            manifest = json.load(open(manifest_path))
        except (ValueError, OSError):
            manifest = {"runs": {}}
    manifest.setdefault("runs", {})

    datasets = discover_datasets(args.data_root, args.image_dir, args.mask_dir, name_filter)
    if not datasets:
        print(f"No datasets (subdirs with {args.image_dir}/ and {args.mask_dir}/) "
              f"found under {args.data_root}")
        return

    print("=" * 70)
    print(f" Gate batch | root={args.data_root} | seeds={seeds} | ks={ks} | "
          f"random_k={args.random_k} | concrete={args.concrete}")
    print(f" Found {len(datasets)} datasets:")
    for ds in datasets:
        rng = f"{ds['range'][0]}-{ds['range'][1]}" if ds["range"] else "?"
        print(f"   - {ds['name']}: C={ds['C']}, n={ds['n_samples']}, range={rng}")
    print("=" * 70)

    specs = build_specs(datasets, seeds, ks, args.random_k, args.concrete)
    print(f"Total training runs planned: {len(specs)}")

    if args.dry_run:
        for sp in specs:
            print(f"  {run_id(sp['ds']['name'], sp['mode'], sp['k'], sp['seed'])}")
        return

    if not args.plots_only:
        append_log(log_path, f"=== batch start: {len(specs)} runs ===")
        done_per_dataset = {ds["name"]: 0 for ds in datasets}
        for i, sp in enumerate(specs):
            status, metrics = run_training(
                sp, base_cfg, args.out_root, args.epochs,
                args.image_dir, args.mask_dir, args.python, args.device, log_path)
            rid = run_id(sp["ds"]["name"], sp["mode"], sp["k"], sp["seed"])
            manifest["runs"][rid] = {
                "status": status, "dataset": sp["ds"]["name"], "mode": sp["mode"],
                "k": sp["k"], "seed": sp["seed"], "C": sp["ds"]["C"], **metrics,
            }
            save_json_atomic(manifest_path, manifest)
            # Refresh this dataset's plots once all its runs are processed.
            done_per_dataset[sp["ds"]["name"]] += 1
            ds_specs = sum(1 for s in specs if s["ds"]["name"] == sp["ds"]["name"])
            if done_per_dataset[sp["ds"]["name"]] == ds_specs:
                plot_dataset_comparison(sp["ds"], manifest, args.out_root, ks, seeds, args.random_k)
                plot_dataset_selected_bands(sp["ds"], manifest, args.out_root, ks, seeds)
        append_log(log_path, "=== batch finished ===")
    else:
        for ds in datasets:
            plot_dataset_comparison(ds, manifest, args.out_root, ks, seeds, args.random_k)
            plot_dataset_selected_bands(ds, manifest, args.out_root, ks, seeds)

    plot_cross_dataset(datasets, manifest, args.out_root, ks, seeds, summary_k)
    print(f"\nDone. Outputs in {args.out_root} (manifest: {manifest_path})")


if __name__ == "__main__":
    main()
