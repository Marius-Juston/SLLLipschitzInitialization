# Reviewer revision: reproducible experiments

This pipeline is independent of the historical `src/train.py`. It implements the
paper's **column-normalized** feedforward layers, effective q=1 at initialization,
explicit bias modes, and a logit head without ReLU. The legacy layer retains its
row normalization; its uninitialized zero-bias bug is fixed separately.

## Environment (uv)

All revision dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.
Use Python 3.10 and a CUDA-capable Linux host:

```bash
uv sync --frozen
uv run python -m pytest -q
uv run python -m revision.math_checks
uv run python -m revision.runner download
```

On this host, ROS injects unrelated modules through `PYTHONPATH`. Prefix commands
with `env -u PYTHONPATH` to isolate the project, for example:

```bash
env -u PYTHONPATH uv run --frozen python -m pytest -q
```

Add dependencies with **`uv add`**, not pip. The PyTorch CUDA 12.1 index is explicit
and used only for torch/torchvision. AutoAttack and the vendored SLL layers are
pinned to Git commits. `revision/vendor/PROVENANCE.txt` records upstream provenance
and the upstream license is included.

## Pilot and launch

```bash
uv run python -m revision.runner pilot \
  --config configs/revision/cifar10-residual-d20-default-s0.json \
  --output review-runs/pilot-residual --device cuda:0 --steps 20
uv run python -m revision.campaign run --profile full --devices 0 1 2 3
```

The full grid contains 132 runs: six practical residual runs, 36 feedforward
CIFAR-10 runs, and 90 Covertype runs. The compact profile drops feedforward depth
15 and uses three Covertype seeds (57 runs). Choose a profile **before** inspecting
test results. The scheduler refuses to change an existing campaign's profile.
CIFAR-100 is not part of the required campaign.

On the current host the campaign is running as the persistent user service
`lipschitz-review-campaign.service`. Prefer these monitoring/control commands:

```bash
systemctl --user status lipschitz-review-campaign
uv run python -m revision.campaign status
# To stop all workers safely at their last complete epoch checkpoint:
systemctl --user stop lipschitz-review-campaign
# To resume this existing service:
systemctl --user start lipschitz-review-campaign
```

The service writes `review-runs/campaign.log` and survives the tool session. The
initial detached shell process was cleaned up by the tool runtime, so `nohup`
alone should not be relied on when launching through that runtime. For a normal
interactive shell (rather than this tool runtime), the alternative is:

```bash
mkdir -p review-runs
nohup env -u PYTHONPATH uv run --frozen python -m revision.campaign run \
  --profile full --devices 0 1 2 3 > review-runs/campaign.log 2>&1 < /dev/null &
echo $!
```

This survives the shell session. It does not promise to wake an assistant. A
filesystem lock prevents duplicate schedulers. Do not launch a second campaign
against the same output directory. Stop the scheduler and its child processes
before resuming after an interruption; preserve checkpoints.

## Monitor and resume

```bash
uv run python -m revision.campaign status
cat review-runs/progress.json
uv run tensorboard --logdir review-runs --host 127.0.0.1 --port 6006
```

Each run has `train.log`, `history.json`, `status.json`, TensorBoard events,
`last.pt`, `best.pt`, parameter/source provenance, and initialization/final layer
diagnostics. Checkpoints include optimizer and RNG states. The validation split
and per-epoch data-order seeds are fixed. Resuming the same campaign reuses
checkpoints and skips completed evaluations:

```bash
uv run python -m revision.campaign run --profile full --devices 0 1 2 3
```

Failed jobs remain visible and the scheduler exits nonzero if any fail. On a new
invocation, failed training resumes from its latest checkpoint. A failed evaluation
can rerun after loading a finished training checkpoint. Do not edit study code
while the campaign runs; provenance records revisions and hashes.

## Protocol

- Covertype: split 72/8/20% with seed 0, train-only standardization; hidden width
  64, depths 5/15/30; Adam 1e-3, batch 64, 25 epochs. No image-noise/adversarial
  metrics are applied to its categorical inputs.
- CIFAR-10 feedforward: fixed 45,000/5,000 training/validation split, standard
  crop/flip augmentation; widths 256, depths 5/15/30, three paired seeds;
  Adam 1e-3, batch 128, 100 epochs. Effective q is initially one in both AOL and
  SLL, fixed for AOL and trainable for SLL.
- Residual control: official small SLL architecture (20 convolutional residual
  blocks, seven dense residual blocks), 1,000 epochs, Adam 0.01 with betas
  (0.5,0.9), upstream triangular LR and temperature/offset loss, batch 64, upstream
  color augmentation. The harness adds a validation split/checkpoint selection.
  The corrected-bias ablation is a **heuristic transfer** of the feedforward
  formula to internal affine maps, not a residual-theory prediction. Upstream
  default q and head initialization are retained in both residual conditions.
- Bias conditions use separate random generators so subsequent feedforward
  weights are paired; bias samples differ across layers.
- Certificates: vector bound 1 for feedforward logits; pairwise head-row distances
  over a 1-Lipschitz residual backbone. Mean subtraction does not change the
  input norm. Certified accuracy uses correct true-class predictions and strict
  radius comparisons, with raw pixel coordinates [0,1].
- Full-test clean and certified accuracy at l2 radii 0.25/0.5/1.0; full-test Gaussian
  noise accuracy at standard deviations 0.01/0.03/0.05. Noise is clipped to [0,1].
- PGD: 100 steps, five restarts, step size 2.5 epsilon / steps, projected l2 ball
  and image box, retaining all successful iterates; fixed seed-independent 1,000
  test examples. Residual models additionally use standard l2 AutoAttack on the
  same examples. Parameter-only normalization is cached for evaluation without
  changing input gradients.

## Completed campaign and result validation

All 132 registered runs completed training and evaluation in 37.25 elapsed hours.
The analysis found no consistent robustness benefit from variance-corrected bias.
See `artifacts/campaign/ANALYSIS.md` for interpretation and validation limits.

Reproduce the audits and manuscript assets from the retained local run directory:

```bash
env -u PYTHONPATH uv run --frozen python -m revision.analyze_results
env -u PYTHONPATH uv run --frozen python -m revision.replay_results --device cuda:0
env -u PYTHONPATH uv run --frozen python -m revision.precision_diagnostics
env -u PYTHONPATH uv run --frozen python -m revision.render_results
```

`analyze_results` checks configurations, contiguous histories, checkpoint selection,
recorded source versions, margin-derived certificates, and same-subset attack bounds.
`replay_results` reloads every best checkpoint, recomputes full-test clean accuracy
and CIFAR certificates, and checks every PGD iterate on eight examples per image
model. `precision_diagnostics` uses cuda:1 for a float64 initialization diagnostic;
it does not retrain any model or modify the original float32 results.

The versioned artifact directory contains per-seed CSVs, paired differences,
compressed evaluation/history/diagnostic/provenance records, per-example margins,
source hashes, replay validation, tables, and the figure. Large checkpoints remain
local in `review-runs/` and are not embedded in Git. The original generic
`revision.campaign report` remains available; the paper uses `result-tables.tex`
from the audited renderer rather than its preliminary all-in-one table.

The deterministic mathematical checks use independent matrices as sampling units.
Small-dimensional quadrature uses an adaptive reference integration; large
matrix dimensions also compare two Gauss-Laguerre orders. The Mathematica file
under `symbolic/` documents the corrected chi-distribution integrand. It has not
been executed by the Python tests.
