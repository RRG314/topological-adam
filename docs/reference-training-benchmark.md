# Reference training benchmark

This benchmark is the minimum real-data training example intended for JOSS
reviewers. It is not a broad optimizer study. Its purpose is to show that the
package can be installed, used in an ordinary PyTorch training loop, tuned under
a documented protocol, and reproduced locally on a dataset that does not require
network access.

## Protocol

- Dataset: `sklearn.datasets.load_digits` real handwritten digit data.
- Model: `Linear(64, 64)`, ReLU, `Linear(64, 10)`.
- Metric: held-out cross-entropy and held-out accuracy.
- Optimizers: Adam, AdamW, `TopologicalAdamV3`, and `TopologicalAdamV4`.
- Learning-rate grid: `3e-4`, `1e-3`, `3e-3`, `1e-2`.
- Tuning seeds: `0`, `1`, `2`.
- Fresh evaluation seeds: `5`, `6`, `7`, `8`, `9`.
- Selection rule: lowest mean held-out cross-entropy over tuning seeds.
- Stored output: `reference_training_results.json`.

Run the stored protocol:

```bash
python examples/reference_training_benchmark.py --out reference_training_results.json
```

Run a quick smoke version:

```bash
python examples/reference_training_benchmark.py --quick --out tmp/reference_training_quick.json
```

## Stored result

The current stored output reports the following fresh-seed summaries:

| Optimizer | Tuned LR | Held-out CE median | Held-out CE mean | Accuracy median | Accuracy mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adam | 1e-2 | 0.1095 | 0.1120 | 0.9733 | 0.9738 |
| AdamW | 1e-2 | 0.1071 | 0.1094 | 0.9733 | 0.9733 |
| V3 | 1e-2 | 0.0965 | 0.1043 | 0.9733 | 0.9738 |
| V4 | 1e-2 | 0.1063 | 0.1093 | 0.9733 | 0.9724 |

Interpretation: this benchmark supports the practical installation and
ordinary-model-training claim. It does not claim broad optimizer superiority.
On this small real-data classification task, all optimizers reach similar
accuracy. V3 has the best median held-out cross-entropy in this run, while V4
behaves near Adam as expected for ordinary stochastic classification.

The script also records per-seed losses, accuracies, timing, tuned learning
rates, and paired deltas against Adam in `reference_training_results.json`.
