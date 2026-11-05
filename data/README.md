# Data

The repository includes the numerical reference solutions and IRK coefficients from [maziarraissi/PINNs](https://github.com/maziarraissi/PINNs), commit `932f50a2d8ef4e80d1456bbae6887a73ff5166ef`.

| File (under `third_party/raissi/`) | Contents |
|---|---|
| `main/Data/NLS.mat` | Complex Schrodinger solution, 256 spatial points and 201 times |
| `main/Data/AC.mat` | Allen-Cahn solution, 512 spatial points and 201 times |
| `Utilities/IRK_weights/Butcher_IRK100.txt` | 100-stage IRK weights and nodes |

`manifest.json` records each source URL, file size, and SHA-256. The original TensorFlow examples and MIT license are retained in `third_party/raissi/`.

Verify the included files with `python scripts/verify_data.py`. Add `--download` to restore files from the pinned upstream version.
