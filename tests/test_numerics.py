from pathlib import Path
import copy
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
from pinn_main.model import PINN
from pinn_main.config import load_config
from pinn_main.data import load_reference, load_irk, sample_training
from pinn_main.physics import schrodinger_residual, componentwise_dx, allen_cahn_initial
from pinn_main.problems import Problem
from pinn_main.optim import ScipyObjective, TF1Adam


def small_config(model):
    config = load_config(ROOT / f"configs/{model}_smoke.json")
    config.update(n_initial=5, layers=[2, 4, 2] if model == "schrodinger" else [1, 4, 101])
    if model == "schrodinger":
        config.update(n_boundary=4, n_residual=11)
    return config


class MainModelNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reference_orientation_and_initial_conditions(self):
        nls, ac = load_reference("schrodinger"), load_reference("allen_cahn")
        self.assertEqual(nls.solution.shape, (201, 256))
        self.assertEqual(ac.solution.shape, (201, 512))
        np.testing.assert_allclose(nls.solution[0], 2 / np.cosh(nls.x), atol=1e-12)
        np.testing.assert_allclose(ac.solution[0], ac.x**2 * np.cos(np.pi * ac.x), atol=1e-12)
        np.testing.assert_allclose(ac.t[[20, 180]], [0.1, 0.9])

    def test_training_does_not_read_future_reference_labels(self):
        for model in ("schrodinger", "allen_cahn"):
            config = small_config(model)
            ref = load_reference(model)
            original = sample_training(ref, config)
            training_time = 0 if model == "schrodinger" else config["initial_time_index"]
            modified = copy.deepcopy(ref)
            mask = np.arange(len(ref.t)) != training_time
            modified.solution[mask] = 999
            changed = sample_training(modified, config)
            for key in original:
                np.testing.assert_array_equal(original[key], changed[key])

    def test_nls_exact_plane_wave(self):
        amplitude, wave_number = 0.7, 1.3
        omega = 0.5 * wave_number**2 - amplitude**2

        class PlaneWave(torch.nn.Module):
            def forward(self, xt):
                phase = wave_number * xt[:, :1] - omega * xt[:, 1:2]
                return amplitude * torch.cat((phase.cos(), phase.sin()), 1)

        points = torch.tensor([[-0.9, 0.2], [0.3, 0.7], [1.1, 0.9]], dtype=torch.float64)
        self.assertLess(
            float(schrodinger_residual(PlaneWave(), points).detach().abs().max()), 1e-12
        )

    def test_nls_polynomial_real_and_imaginary_signs(self):
        class Polynomial(torch.nn.Module):
            def forward(self, xt):
                x, t = xt[:, :1], xt[:, 1:2]
                return torch.cat((x**2 + t, x**3 - t**2), 1)

        points = torch.tensor([[-0.4, 0.1], [0.7, 0.3]], dtype=torch.float64)
        x, t = points[:, :1], points[:, 1:2]
        u, v = x * x + t, x**3 - t * t
        expected = torch.cat((1 + 3 * x + (u * u + v * v) * v, -2 * t - 1 - (u * u + v * v) * u), 1)
        torch.testing.assert_close(
            schrodinger_residual(Polynomial(), points), expected, rtol=1e-12, atol=1e-12
        )

    def test_all_stage_derivatives_match_separate_output_autograd(self):
        torch.manual_seed(6)
        model = PINN([1, 5, 4], [-1], [1], torch.float64)
        x = torch.tensor([[-0.8], [0.1], [0.6]], dtype=torch.float64, requires_grad=True)
        y = model(x)
        first = componentwise_dx(y, x)
        second = componentwise_dx(first, x)
        first_expected, second_expected = [], []
        for column in range(y.shape[1]):
            dx = torch.autograd.grad(y[:, column].sum(), x, create_graph=True)[0]
            dxx = torch.autograd.grad(dx.sum(), x, create_graph=True)[0]
            first_expected.append(dx)
            second_expected.append(dxx)
        torch.testing.assert_close(first, torch.cat(first_expected, 1), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(second, torch.cat(second_expected, 1), rtol=1e-12, atol=1e-12)

    def test_irk_orientation_and_pde_sign_with_manufactured_stages(self):
        root3 = np.sqrt(3)
        coefficients = torch.tensor(
            [[0.25, 0.25 - root3 / 6], [0.25 + root3 / 6, 0.25], [0.5, 0.5]], dtype=torch.float64
        )

        class Stages(torch.nn.Module):
            def forward(self, x):
                return torch.cat((x * x + x**3, 2 * x * x - x**3, x * x + 0.3), 1)

        x = torch.tensor([[-0.4], [0.2]], dtype=torch.float64)
        first, second = x * x + x**3, 2 * x * x - x**3
        f1 = 5 * first - 5 * first**3 + 0.0001 * (2 + 6 * x)
        f2 = 5 * second - 5 * second**3 + 0.0001 * (4 - 6 * x)
        expected = Stages()(x) - 0.8 * torch.cat((f1, f2), 1) @ coefficients.T
        actual = allen_cahn_initial(Stages(), x, 0.8, coefficients)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_official_irk_coefficients_integrate_monomials(self):
        weights, nodes = load_irk()
        for power in range(5):
            np.testing.assert_allclose(
                weights[:-1] @ nodes**power, nodes ** (power + 1) / (power + 1), atol=1e-12
            )
            self.assertAlmostEqual(float(weights[-1] @ nodes**power), 1 / (power + 1), places=12)

    def test_chunked_losses_and_parameter_gradients_equal_full_batch(self):
        for name in ("schrodinger", "allen_cahn"):
            torch.manual_seed(17)
            problem = Problem(small_config(name))
            full = problem.backward(1000)
            gradient = [p.grad.clone() for p in problem.model.parameters()]
            chunked = problem.backward(3)
            for key in full:
                self.assertAlmostEqual(full[key], chunked[key], places=9)
            for p, expected in zip(problem.model.parameters(), gradient):
                torch.testing.assert_close(p.grad, expected, rtol=1e-10, atol=1e-10)

    def test_full_parameter_gradients_against_finite_differences(self):
        for name in ("schrodinger", "allen_cahn"):
            torch.manual_seed(19)
            problem = Problem(small_config(name))
            objective = ScipyObjective(problem)
            values = objective.pack()
            _, gradient = objective(values)
            for index in [0, 3, len(values) // 2, len(values) - 1]:
                plus, minus = values.copy(), values.copy()
                plus[index] += 1e-6
                minus[index] -= 1e-6
                estimate = (objective(plus)[0] - objective(minus)[0]) / (2e-6)
                np.testing.assert_allclose(gradient[index], estimate, rtol=2e-5, atol=2e-6)

    def test_tf1_adam_update_and_restored_moments(self):
        value = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        optimizer = TF1Adam([value])
        expected, m, v = 1.0, 0.0, 0.0
        for step, grad in enumerate([0.2, -0.3, 0.1], 1):
            value.grad = torch.tensor([grad], dtype=torch.float64)
            optimizer.step()
            m, v = 0.9 * m + 0.1 * grad, 0.999 * v + 0.001 * grad * grad
            expected -= 0.001 * np.sqrt(1 - 0.999**step) / (1 - 0.9**step) * m / (np.sqrt(v) + 1e-8)
            self.assertAlmostEqual(value.item(), expected, places=14)
        other = torch.nn.Parameter(value.detach().clone())
        resumed = TF1Adam([other])
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        for opt, parameter in ((optimizer, value), (resumed, other)):
            parameter.grad = torch.tensor([0.6], dtype=torch.float64)
            opt.step()
        torch.testing.assert_close(value, other, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
