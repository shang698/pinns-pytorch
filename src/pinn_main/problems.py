import numpy as np
import torch

from .model import PINN
from .data import load_reference, load_irk, sample_training
from .physics import nls_fields, schrodinger_residual, allen_cahn_initial, allen_cahn_boundary


class Problem:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.loss_weights = dict(config["loss_weights"])
        self.dtype = {"float32": torch.float32, "float64": torch.float64}[config["dtype"]]
        self.device = device
        self.reference = load_reference(config["model"])
        self.arrays = sample_training(self.reference, config)
        self.batch = {
            k: self.tensor(v) for k, v in self.arrays.items() if not k.endswith("indices")
        }
        if config["model"] == "schrodinger":
            lower, upper = [-5.0, 0.0], [5.0, np.pi / 2]
            self.loss_keys = ("data", "boundary", "boundary_derivative", "physics")
        else:
            lower, upper = [-1.0], [1.0]
            q = config["irk_stages"]
            self.loss_keys = ("irk", "boundary", "boundary_derivative")
            if "irk" not in self.loss_weights:
                self.loss_weights["irk"] = self.loss_weights["data"]
            weights, nodes = load_irk(q)
            self.weights = self.tensor(weights)
            self.arrays.update(
                irk_weights=weights.astype(
                    np.float32 if self.dtype == torch.float32 else np.float64
                ),
                irk_nodes=nodes,
            )
        self.model = PINN(config["layers"], lower, upper, self.dtype, device)

    def tensor(self, array):
        return torch.as_tensor(array, dtype=self.dtype, device=self.device)

    def backward(self, chunk_size=None):
        chunk_size = chunk_size or self.config["chunk_size"]
        self.model.zero_grad(set_to_none=True)
        losses = dict.fromkeys(self.loss_keys, 0.0)
        weights = self.loss_weights

        def add(name, value):
            (weights[name] * value).backward()
            losses[name] += float(value.detach())

        b = self.batch
        if self.config["model"] == "schrodinger":
            # Sum the real and imaginary MSEs.
            add(
                "data",
                (self.model(b["initial_points"]) - b["initial_values"]).square().sum()
                / len(b["initial_points"]),
            )
            times = b["boundary_times"]
            left = torch.cat((-5 * torch.ones_like(times), times), 1)
            right = torch.cat((5 * torch.ones_like(times), times), 1)
            _, ul, dul, dvl = nls_fields(self.model, left)
            _, ur, dur, dvr = nls_fields(self.model, right)
            boundary = (ul - ur).square().sum() / len(times)
            derivative = (
                (dul[:, :1] - dur[:, :1]).square().sum() + (dvl[:, :1] - dvr[:, :1]).square().sum()
            ) / len(times)
            (
                weights["boundary"] * boundary + weights["boundary_derivative"] * derivative
            ).backward()
            losses["boundary"], losses["boundary_derivative"] = (
                float(boundary.detach()),
                float(derivative.detach()),
            )
            residual = b["residual_points"]
            for points in residual.split(chunk_size):
                add(
                    "physics",
                    schrodinger_residual(self.model, points).square().sum() / len(residual),
                )
        else:
            for points, target in zip(
                b["initial_points"].split(chunk_size), b["initial_values"].split(chunk_size)
            ):
                reconstructed = allen_cahn_initial(self.model, points, b["dt"], self.weights)
                add("irk", (reconstructed - target).square().sum())
            stages, dx = allen_cahn_boundary(self.model, b["boundary_points"])
            boundary = (stages[0] - stages[1]).square().sum()
            derivative = (dx[0] - dx[1]).square().sum()
            (
                weights["boundary"] * boundary + weights["boundary_derivative"] * derivative
            ).backward()
            losses["boundary"], losses["boundary_derivative"] = (
                float(boundary.detach()),
                float(derivative.detach()),
            )
        losses["loss"] = sum(weights[k] * losses[k] for k in self.loss_keys)
        if not np.isfinite(list(losses.values())).all():
            raise FloatingPointError("Non-finite loss")
        return losses

    def evaluate(self):
        ref = self.reference
        with torch.no_grad():
            if self.config["model"] == "schrodinger":
                xx, tt = np.meshgrid(ref.x, ref.t)
                points = self.tensor(np.column_stack((xx.ravel(), tt.ravel())))
                raw = np.concatenate(
                    [self.model(part).cpu().numpy() for part in points.split(2048)]
                )
                prediction = (
                    raw[:, 0].astype(np.float64) + 1j * raw[:, 1].astype(np.float64)
                ).reshape(ref.solution.shape)
                reference = ref.solution
                relative = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
                metrics = {
                    "relative_l2_complex": relative(prediction, reference),
                    "relative_l2_amplitude": relative(abs(prediction), abs(reference)),
                    "relative_l2_real": relative(prediction.real, reference.real),
                    "relative_l2_imaginary": relative(prediction.imag, reference.imag),
                    "relative_l2_complex_after_initial_time": relative(
                        prediction[1:], reference[1:]
                    ),
                }
                saved = {"x": ref.x, "t": ref.t, "reference": reference, "prediction": prediction}
            else:
                stages = self.model(self.tensor(ref.x[:, None])).cpu().numpy().astype(np.float64)
                prediction = stages[:, -1]
                reference = ref.solution[self.config["target_time_index"]].real
                metrics = {
                    "relative_l2_target_snapshot": float(
                        np.linalg.norm(prediction - reference) / np.linalg.norm(reference)
                    ),
                    "initial_time": float(self.arrays["initial_time"]),
                    "target_time": float(self.arrays["target_time"]),
                }
                saved = {
                    "x": ref.x,
                    "reference": reference,
                    "prediction": prediction,
                    "internal_stage_predictions": stages[:, :-1],
                    "internal_stage_times": self.arrays["initial_time"]
                    + self.arrays["dt"] * self.arrays["irk_nodes"],
                    "initial_time": self.arrays["initial_time"],
                    "target_time": self.arrays["target_time"],
                }
        metrics.update(
            rmse=float(np.sqrt(np.mean(abs(prediction - reference) ** 2))),
            max_absolute_error=float(np.max(abs(prediction - reference))),
        )
        return metrics, saved
