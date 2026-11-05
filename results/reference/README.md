# Reference runs

Six completed runs used in the report:

- `schrodinger/baseline` and `allen_cahn/baseline`: float32 training with the paper configurations.
- `*/float32` and `*/float64`: independent continuations of the corresponding baseline.

Each folder includes the configuration, final model, predictions, metrics, training points, loss history, and software/GPU information. Continuation folders also include their starting checkpoint. The two arms share the baseline weights and sampled points; IRK coefficients use each arm's dtype.

Run `python scripts/evaluate.py --reference` to check all saved models on CPU. Use `python scripts/plot_results.py` to redraw the figures.

These runs predate the public-code cleanup. Their configurations, protocol names, weights, and histories are kept as recorded. Current commands use shorter protocol names and the same numerical settings. In the original Allen-Cahn histories, `data` is the IRK reconstruction loss and `physics` is zero; new runs name that term `irk`.

Environment records retain the original software versions and source hashes. Machine-specific paths and job identifiers were omitted from the public copy. Each continuation starts from `../baseline/model.pt`.
