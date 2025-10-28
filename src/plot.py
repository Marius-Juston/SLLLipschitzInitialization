#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

# Third-party (ships with TensorBoard / TensorFlow installations)
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    sys.stderr.write(
        "ERROR: Could not import TensorBoard's EventAccumulator. "
        "Please `pip install tensorboard` (or TensorFlow) on your machine.\n"
    )
    raise

DPI = 600

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Data containers and parsing
# -----------------------------
@dataclass(frozen=True)
class RunMeta:
    name: str
    lr: str
    n_layers: int
    bias_init: bool
    sample: str
    run_dir: str
    run_id: str  # unique id (e.g., the folder name)


RUN_DIR_RE = re.compile(
    r"""^(?P<name>.+?)                 # experiment name (greedy, minimal)
        _lr(?P<lr>[^_]+)              # learning rate right after _lr
        _n(?P<n_layers>\d+)           # number of layers right after _n
        _b(?P<bias_init>True|False)   # bias flag right after _b
        _(?P<sample>.+)$              # sample id (rest)
    """,
    re.X
)

BETTER_NAME = {
    'loss/train' : 'Training Loss',
    'loss/test' : 'Testing Loss',
    'loss/correct' : 'Accuracy'
}


def parse_run_dir(dir_name: str, full_path: str) -> Optional[RunMeta]:
    m = RUN_DIR_RE.match(dir_name)
    if not m:
        return None
    try:
        return RunMeta(
            name=m.group("name"),
            lr=m.group("lr"),
            n_layers=int(m.group("n_layers")),
            bias_init=(m.group("bias_init") == "True"),
            sample=m.group("sample"),
            run_dir=full_path,
            run_id=dir_name
        )
    except Exception:
        return None


def discover_runs(runs_root: str) -> List[RunMeta]:
    metas: List[RunMeta] = []
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"runs_dir does not exist or is not a directory: {runs_root}")
    for entry in os.listdir(runs_root):
        full = os.path.join(runs_root, entry)
        if not os.path.isdir(full):
            continue
        meta = parse_run_dir(entry, full)
        if meta is None:
            # skip folders that don't match the naming pattern
            continue
        # make sure there is at least one event file inside
        evts = glob.glob(os.path.join(full, "events.*")) + glob.glob(os.path.join(full, "*tfevents*"))
        if not evts:
            # some loggers create a subdir; allow scanning 1-level deep
            subdirs = [os.path.join(full, d) for d in os.listdir(full) if os.path.isdir(os.path.join(full, d))]
            found = False
            for sd in subdirs:
                evts = glob.glob(os.path.join(sd, "events.*")) + glob.glob(os.path.join(sd, "*tfevents*"))
                if evts:
                    found = True
                    break
            if not found:
                continue
        metas.append(meta)
    return metas


# -----------------------------
# Loading scalar data
# -----------------------------
SCALAR_TAGS = ["loss/train", "loss/test", "loss/correct"]


def load_scalars_for_run(meta: RunMeta, tags: List[str]) -> pd.DataFrame:
    """
    Load specified scalar tags for a single run directory using EventAccumulator.
    Returns a DataFrame with columns:
        run_id, name, lr, n_layers, bias_init, sample, tag, step, wall_time, value
    """
    # Try to build an EventAccumulator over the run dir; EA will recurse to merge multiple event files
    # Set size guidance to load *all* scalar points
    ea = EventAccumulator(meta.run_dir, size_guidance={"scalars": 0})
    ea.Reload()  # parse protobuf files

    available = set(ea.Tags().get("scalars", []))
    rows = []
    for t in tags:
        if t not in available:
            # skip missing tags
            continue
        for ev in ea.Scalars(t):
            rows.append({
                "run_id": meta.run_id,
                "name": meta.name,
                "lr": meta.lr,
                "n_layers": meta.n_layers,
                "bias_init": meta.bias_init,
                "sample": meta.sample,
                "tag": t,
                "step": ev.step,
                "wall_time": ev.wall_time,
                "value": float(ev.value),
            })
    return pd.DataFrame(rows)


def load_all(runs_root: str, tags: List[str]) -> pd.DataFrame:
    metas = discover_runs(runs_root)
    print("Loaded", len(metas), "runs.")

    if not metas:
        raise RuntimeError(f"No matching runs found under: {runs_root}")
    frames = []
    skipped = 0
    for m in metas:
        try:
            df = load_scalars_for_run(m, tags)
            if not df.empty:
                frames.append(df)
            else:
                skipped += 1
        except Exception as e:
            skipped += 1
            sys.stderr.write(f"[warn] failed to read {m.run_dir}: {e}\n")
    if not frames:
        raise RuntimeError("Found runs, but none had the requested scalar tags.")
    out = pd.concat(frames, axis=0, ignore_index=True)
    out.sort_values(["tag", "run_id", "step"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    if skipped:
        sys.stderr.write(f"[info] Loaded {len(frames)} runs; skipped {skipped} with missing/no data.\n")
    return out


# -----------------------------
# Plotting helpers (matplotlib-only)
# -----------------------------
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_loss_colored_by_layers(df: pd.DataFrame, outdir: str, tag: str = "loss/train"):
    sub = df[df["tag"] == tag].copy()
    if sub.empty:
        sys.stderr.write(f"[warn] No data for tag '{tag}', skipping first plot.\n")
        return
    # Map n_layers -> a color (use the default color cycle)
    layers = sorted(sub["n_layers"].unique().tolist())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    layer_to_color = {nl: color_cycle[i % len(color_cycle)] for i, nl in enumerate(layers)}

    plt.figure(figsize=(9, 6))
    for rid, g in sub.groupby("run_id"):
        c = layer_to_color[g["n_layers"].iloc[0]]
        plt.plot(g["step"], g["value"], color=c)
    # legend keys for layers
    handles = []
    labels = []
    for nl in layers:
        handles.append(plt.Line2D([0], [0], color=layer_to_color[nl], lw=2))
        labels.append(f"n_layers={nl}")
    plt.legend(handles, labels, title="Layers", loc="best")
    plt.xlabel("Epoch")
    plt.ylabel(BETTER_NAME[tag])
    plt.yscale('log')
    plt.title(f"{BETTER_NAME[tag]} vs Number of Layers")
    plt.grid(True, alpha=0.3)
    out = os.path.join(outdir, f"{tag.replace('/', '_')}_by_layers.png")
    plt.tight_layout()
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"[saved] {out}")


def plot_grid_by_layers_bias(df: pd.DataFrame, outdir: str, tag: str = "loss/train"):
    """
    Create a grid with rows = [5, 15, 30], cols = [bias=True, bias=False].
    Each panel shows individual *samples* (no aggregation) for the chosen tag.
    """
    rows = [5, 15, 30]
    cols = [False, True]

    sub = df[(df["tag"] == tag) & (df["n_layers"].isin(rows))].copy()
    if sub.empty:
        sys.stderr.write(f"[warn] No data for tag '{tag}' among n_layers {rows}, skipping grid.\n")
        return

    fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols), figsize=(12, 9), sharex=True, sharey=True)
    if len(rows) == 1 and len(cols) == 1:
        axes = [[axes]]

    for i, nl in enumerate(rows):
        for j, b in enumerate(cols):
            ax = axes[i][j]
            cell = sub[(sub["n_layers"] == nl) & (sub["bias_init"] == b)]
            # One line per run_id (sample)

            for rid, g in cell.groupby("run_id"):
                # create a concise label per sample run
                # sample_label = g["sample"].iloc[0]
                # lr = g["lr"].iloc[0]
                # ax.plot(g["step"], g["value"], linewidth=1.1, alpha=0.9, label=f"{sample_label} (lr={lr})")
                ax.plot(g["step"], g["value"], c='tab:red' if b else 'tab:blue')

            steps = cell.groupby('step')
            means = steps['value'].mean()
            std = steps['value'].std()

            line, = ax.plot(means.index, means, label="Mean", c='k')
            fill = ax.fill_between(means.index, means - std, means + std, alpha=0.25, color='r' if b else 'b')
            ax.legend(handles=[(line, fill)], labels=["Mean ± Std"], loc="best")

            # Overall statistics

            ax.set_title(f"n_layers={nl}, bias={b}")
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            if i == len(rows) - 1:
                ax.set_xlabel("Epoch")
            if j == 0:
                ax.set_ylabel(BETTER_NAME[tag])
            # keep legends lean by limiting to first few labels if too many
            # if cell["run_id"].nunique() <= 10:
            #     ax.legend(fontsize=8, loc="best")
            # else:
            #     ax.legend([], [], frameon=False)

    # fig.suptitle(f"{BETTER_NAME[tag]} per-sample\nRows: n_layers in {rows} | Cols: bias_init in {cols}", y=0.995)
    fig.suptitle(f"{BETTER_NAME[tag]} per-sample")
    out = os.path.join(outdir, f"{tag.replace('/', '_')}_grid_layers_bias.png")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[saved] {out}")


def compute_trained_flags(df: pd.DataFrame, rel_threshold: float) -> pd.DataFrame:
    """
    For each run_id, decide if 'trained' based on absolute change between first and last
    training loss exceeding a small relative threshold:
        |last - first| > rel_threshold * max(1, |first|)
    Returns a DataFrame with columns:
        run_id, n_layers, bias_init, sample, lr, first, last, delta, trained (bool)
    """
    tag = "loss/train"
    sub = df[df["tag"] == tag]
    rows = []
    for rid, g in sub.groupby("run_id"):
        g = g.sort_values("step")
        if len(g) < 2:
            continue
        first = float(g["value"].iloc[5])
        last = float(g["value"].iloc[-1])
        delta = abs(last - first)
        thresh = rel_threshold
        trained = (delta > thresh)
        rows.append({
            "run_id": rid,
            "n_layers": int(g["n_layers"].iloc[0]),
            "bias_init": bool(g["bias_init"].iloc[0]),
            "sample": str(g["sample"].iloc[0]),
            "lr": str(g["lr"].iloc[0]),
            "first": first,
            "last": last,
            "delta": delta,
            "threshold": thresh,
            "trained": trained
        })
    return pd.DataFrame(rows)


def plot_trained_summary(status_df: pd.DataFrame, outdir: str):
    """
    Show a grouped stacked bar chart:
      - x-axis: n_layers groups (e.g., 5, 15, 30, ... if present)
      - within each group: two bars (bias=False and bias=True)
      - each bar stacked with counts of Not Trained (bottom) and Trained (top)
    """
    if status_df.empty:
        sys.stderr.write("[warn] No status data to plot.\n")
        return

    layers = sorted(status_df["n_layers"].unique().tolist())
    biases = [False, True]

    # compute counts
    counts = {}
    for nl in layers:
        counts[nl] = {}
        for b in biases:
            sub = status_df[(status_df["n_layers"] == nl) & (status_df["bias_init"] == b)]
            n_trained = int((sub["trained"] == True).sum())
            n_not = int((sub["trained"] == False).sum())
            counts[nl][b] = (n_trained, n_not)

    # plotting
    plt.figure(figsize=(10, 6))
    x_centers = list(range(len(layers)))
    width = 0.35  # bar width for bias groups

    for i, b in enumerate(biases):
        xs = [xc + (i - 0.5) * width for xc in x_centers]  # offset for bias
        trained_vals = [counts[nl][b][0] for nl in layers]
        not_vals = [counts[nl][b][1] for nl in layers]
        # bottom stack: not trained
        plt.bar(xs, not_vals, width=width, label=f"bias={b} - Not Trained")
        # top stack: trained
        plt.bar(xs, trained_vals, width=width, bottom=not_vals, label=f"bias={b} - Trained", alpha=0.8)

    plt.xticks(x_centers, [str(nl) for nl in layers])
    plt.xlabel("n_layers")
    plt.ylabel("Run count")
    plt.title("Training status by Number of Layers and Bias\n(Trained = |last-loss − first-loss| exceeds threshold)")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(axis="y", alpha=0.3)
    out = os.path.join(outdir, "trained_status_by_layers_bias.png")
    plt.tight_layout()
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"[saved] {out}")


def main():
    parser = argparse.ArgumentParser(description="Extract TensorBoard scalars and generate requested plots.")
    parser.add_argument("--runs_dir", type=str, required=True, help="Path to runs/ containing your event dirs.")
    parser.add_argument("--threshold", type=float, default=1e-2,
                        help="Relative threshold to deem 'trained' (default: 1e-4).")
    parser.add_argument("--outdir", type=str, default="tb_plots", help="Where to save CSV and figures.")
    parser.add_argument("--tags", type=str, default="loss/train,loss/test,loss/correct",
                        help="Comma-separated scalar tags to extract.")
    args = parser.parse_args()

    ensure_outdir(args.outdir)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    print(f"[info] Scanning runs in: {args.runs_dir}")

    csv_path = os.path.join(args.outdir, "all_scalars.csv")

    if not os.path.exists(csv_path):
        df = load_all(args.runs_dir, tags)
        df.to_csv(csv_path, index=False)
    else:
        print(f"[info] Loading all scalars from {args.runs_dir} ...")
        df = pd.read_csv(csv_path)
    print(f"[saved] {csv_path}  ({len(df)} rows)")

    # 1) Single plot: loss/train colored by n_layers
    plot_loss_colored_by_layers(df, args.outdir, tag="loss/train")

    # 2) Grid: rows=[5,15,30], cols=[bias=True,bias=False], per-sample curves (no aggregation)
    plot_grid_by_layers_bias(df, args.outdir, tag="loss/train")

    # 3) Trained / not-trained status from training loss
    status_df = compute_trained_flags(df, rel_threshold=args.threshold)
    status_csv = os.path.join(args.outdir, "trained_status.csv")
    status_df.to_csv(status_csv, index=False)
    print(f"[saved] {status_csv}")
    plot_trained_summary(status_df, args.outdir)

    print("[done] ✓")


if __name__ == "__main__":
    main()
