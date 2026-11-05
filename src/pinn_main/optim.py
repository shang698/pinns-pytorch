import math
import numpy as np
import torch


class TF1Adam(torch.optim.Optimizer):
    """Match TF1's epsilon placement in the uncorrected second moment."""

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["step"] += 1
                state["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                state["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                scale = group["lr"] * math.sqrt(1 - b2 ** state["step"]) / (1 - b1 ** state["step"])
                p.addcdiv_(state["m"], state["v"].sqrt().add_(group["eps"]), value=-scale)


def packed_gradient(model):
    parameters = list(model.parameters())
    if any(p.grad is None for p in parameters):
        raise RuntimeError("Missing parameter gradient")
    gradient = np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in parameters]).astype(
        np.float64
    )
    if not np.isfinite(gradient).all():
        raise FloatingPointError("Non-finite parameter gradient")
    return gradient


class ScipyObjective:
    def __init__(self, problem, on_evaluation=None):
        self.problem, self.on_evaluation = problem, on_evaluation
        self.parameters = list(problem.model.parameters())
        self.nfev = 0

    def pack(self):
        return np.concatenate([p.detach().cpu().numpy().ravel() for p in self.parameters]).astype(
            np.float64
        )

    def unpack(self, values):
        offset = 0
        with torch.no_grad():
            for p in self.parameters:
                end = offset + p.numel()
                p.copy_(
                    torch.as_tensor(
                        values[offset:end].reshape(p.shape), dtype=p.dtype, device=p.device
                    )
                )
                offset = end

    def __call__(self, values):
        self.unpack(values)
        losses = self.problem.backward()
        gradient = packed_gradient(self.problem.model)
        self.nfev += 1
        if self.on_evaluation:
            self.on_evaluation(self.nfev, losses, float(np.max(abs(gradient))))
        return losses["loss"], gradient
