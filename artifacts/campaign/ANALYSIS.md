# Completed experiment analysis

All 132 registered runs completed their prescribed training and evaluation in
37.25 elapsed hours on four RTX 6000 Ada GPUs. Summed per-epoch training/validation
runtime is 140.31 GPU-hours; this excludes checkpoint writes, initialization
measurements, and final evaluation. These quantities are not GPU kernel timings.

The data do **not** support a consistent benefit from variance-corrected bias.

| Comparison | Corrected minus reference, percentage points (mean ± paired sample SD) |
| --- | ---: |
| Residual CIFAR-10 clean, vs default | −0.44 ± 0.13 |
| Residual certified accuracy, radius 0.5, vs default | +0.39 ± 0.84 |
| Residual AutoAttack accuracy, radius 0.5, vs default | −0.13 ± 0.57 |
| Residual Gaussian noise accuracy, SD 0.05, vs default | −0.75 ± 0.30 |
| Covertype depth 5 clean, vs zero | +0.14 ± 0.25 |
| Covertype depth 15 clean, vs zero | −3.68 ± 7.08 |
| Covertype depth 30 clean, vs zero | −8.97 ± 11.58 |

Residual comparisons use three paired seeds; Covertype uses ten. These SDs are
variability across paired differences, not confidence intervals. No equivalence,
noninferiority, or population-level significance claim follows from these data.

Feedforward CIFAR-10 depth 5 reaches about 44% clean accuracy. Corrected bias
reduces mean clean and certified accuracy for both AOL and SLL. Every depth-15
and depth-30 classifier has 10% clean accuracy and zero certificates at the tested
radii. In Covertype depth 30, four zero-bias seeds exceed 70% accuracy and six
predict only the majority class; all default/corrected seeds predict only that
class. The fixed optimizer and training budget limit conclusions about trainability.

## Validation performed

- All configurations match the 132-job manifest. Histories contain every epoch
  exactly once. The evaluated checkpoint is the earliest maximum of clean
  validation accuracy; no test-driven checkpoint selection was used.
- Original source hashes match each run's recorded Git commit. Four initial runs
  used 52d2f2d and the others used 551d1c1. The only executable difference is a
  status-file write in the optional early-stop/pilot branch, not training or
  evaluation arithmetic. Two runs resumed pilot checkpoints; the available early
  provenance is retained, although those early records lack source hashes.
- Saved 10,000-example margins reproduce certified accuracy and clean correctness
  for every CIFAR model. The exact seed-123 subset was reconstructed: certified
  accuracy ≤ attack accuracy ≤ clean accuracy on the same 1,000 examples for all
  42 CIFAR models and all reported attacks/radii.
- Independent replay loaded all 132 best and last checkpoints, checked finite
  parameter values and recorded epochs, and reproduced full-test clean accuracy.
  All CIFAR per-example correctness and radii were reproduced exactly.
- PGD replay checked all evaluated iterates for valid pixel bounds and l2 norms
  on eight examples per CIFAR model, using the actual 100-step/five-restart attack
  at all three radii. No certified example was broken. This is an implementation
  check, not a second full-subset attack evaluation.
- The original AutoAttack logs report no NaNs, pixel values in [0,1], and valid
  maximum l2 perturbation norms for all six residual models and three radii.
- Float64 initialization diagnostics from the same float32-drawn weights separate
  genuine signal decay from the float32 variance floor. The correction keeps
  activation energy near 0.5 but does not prevent input variance decay.

The original PGD adversarial tensors were not retained. Therefore the audit does
not retrospectively assert per-example attack/certificate consistency over every
original attack tensor; it combines aggregate saved evidence, source inspection,
and the bounded independent replay described above. Checkpoints remain local;
this Git artifact bundle does not include the large weight files.

## Reproducibility artifacts

- `seed-results.csv`: every seed, endpoint, and paired-comparison input.
- `validated-summary.json`: group means/SDs and explicit paired differences.
- `campaign-records.json.gz`: configurations, evaluations, complete training
  histories, initialization/final diagnostics, and provenance for every run.
- `test-margins.npz`: per-example CIFAR radii, correctness, and margins.
- `source-manifest.json`: hashes of local source result files and checkpoints.
- `audit.json`, `checkpoint-replay.json`: machine-readable validation outcomes.
- `precision-diagnostics.json`, `plotted-diagnostics.json`: initialization figure
  inputs, including the numerical-precision check.
- `result-tables.tex`, `initialization-diagnostics.pdf`: generated paper assets.

Reproduction commands are in `REVISION.md`. All scripts use the existing uv
project and lockfile; no additional dependencies or training modifications were
needed for this analysis.
