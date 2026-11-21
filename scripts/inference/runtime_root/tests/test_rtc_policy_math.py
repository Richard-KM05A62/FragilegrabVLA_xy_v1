"""NumPy/AST analytical RTC checks; these do NOT execute JAX or a GPU model.

Only the production arithmetic expressions are loaded, without importing
OpenPI/JAX/Flax. A known analytical denoiser supplies its exact Jacobian in
the sign check; this cannot validate the learned model's VJP or compilation.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from inference.rtc_core import rtc_reference


def named_function(parent: ast.AST, name: str) -> ast.FunctionDef:
    return next(node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name == name)


def compile_expression(node: ast.AST):
    return compile(ast.fix_missing_locations(ast.Expression(node)), "<rtc arithmetic>", "eval")


class RTCAnalyticalMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "inference" / "piper_rtc_policy.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        guidance = named_function(tree, "guidance_weight")
        isolated = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), guidance,
        ], type_ignores=[])
        scope = {"jnp": np}
        exec(compile(ast.fix_missing_locations(isolated), str(source), "exec"), scope)
        cls.guidance = staticmethod(scope["guidance_weight"])
        sampler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RTCSampler")
        integrate = named_function(named_function(sampler, "sample_actions"), "integrate")
        guided = named_function(integrate, "guided")
        denoiser = named_function(guided, "denoiser")
        denoised_return = next(node for node in denoiser.body if isinstance(node, ast.Return))
        velocity_return = next(node for node in guided.body if isinstance(node, ast.Return))
        integration_return = next(node for node in integrate.body if isinstance(node, ast.Return))
        cls.denoised_expression = compile_expression(denoised_return.value.elts[0])
        cls.velocity_expression = compile_expression(velocity_return.value)
        cls.integration_expression = compile_expression(integration_return.value)

    def test_beta_finite_at_both_endpoints_and_bounded(self):
        times = np.array([0, 1e-12, 0.2, 0.5, 0.8, 1-1e-12, 1.0])
        with np.errstate(all="raise"):
            beta = self.guidance(times, 5.0)
        self.assertTrue(np.isfinite(beta).all())
        self.assertTrue(np.all((beta >= 0) & (beta <= 5)))
        self.assertAlmostEqual(float(beta[3]), 2.0)
        np.testing.assert_allclose(beta, beta[::-1], rtol=1e-10)
        np.testing.assert_array_equal(self.guidance(times, 0.0), np.zeros_like(times))

    def test_reversed_time_guidance_moves_toward_reference_for_known_denoiser(self):
        # Exact analytical v(x)=M*x, f(x)=x-t*M*x. This is not a JAX VJP.
        x = np.array([0.8, -0.5, 0.2])
        target = np.array([0.0, 0.1, -0.2])
        matrix = np.diag([0.1, 0.2, 0.3])
        t = 0.5
        v = matrix @ x
        denoised = eval(self.denoised_expression, {"x": x, "t": t, "v": v})
        np.testing.assert_allclose(denoised, x-t*v)
        jacobian = np.eye(3)-t*matrix
        error = target-denoised
        correction = jacobian.T @ error
        settings = SimpleNamespace(max_guidance=5.0, num_steps=10)
        guided_v = eval(self.velocity_expression, {
            "v_t": v, "t": t, "self": settings,
            "guidance_weight": self.guidance, "correction": correction,
        })
        moved = eval(self.integration_expression, {"x_t": x, "v_t": guided_v, "self": settings})
        unguided = x-v/settings.num_steps
        self.assertLess(np.linalg.norm(jacobian @ moved-target), np.linalg.norm(jacobian @ unguided-target))

    def test_initial_parameters_contract_identity_denoiser_error(self):
        # At t=1 the initial beta=5 and dt=-1/10 give a 0.5 correction.
        # This proves the arithmetic for f(x)=x, not stability for Pi0.
        x, target = np.array([1.0, -2.0]), np.array([0.0, 1.0])
        settings = SimpleNamespace(max_guidance=5.0, num_steps=10)
        guided_v = eval(self.velocity_expression, {
            "v_t": np.zeros_like(x), "t": 1.0, "self": settings,
            "guidance_weight": self.guidance, "correction": target-x,
        })
        moved = eval(self.integration_expression, {"x_t": x, "v_t": guided_v, "self": settings})
        np.testing.assert_allclose(moved, (x+target)/2)

    def test_exponential_reference_weights_match_kinetix_schedule(self):
        # Kinetix start=2,end=6,total=10: two full prefix weights, four
        # fractional weights, then zeros. Exercise the production mask builder.
        action = lambda step: np.full(7, step, dtype=np.float32)
        previous = SimpleNamespace(origin_step=-4, action_at=action)
        executor = SimpleNamespace(action_at=action)
        _, weights = rtc_reference(0, 2, previous, executor, SimpleNamespace(horizon=10))
        fractions = np.array([4/5, 3/5, 2/5, 1/5], dtype=np.float32)
        expected = np.concatenate([np.ones(2), fractions*np.expm1(fractions)/(math.e-1), np.zeros(4)])
        np.testing.assert_allclose(weights, np.repeat(expected[:, None], 7, axis=1), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
