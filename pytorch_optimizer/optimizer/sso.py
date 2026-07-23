from typing import Optional, Tuple, Union

import torch
from torch.nn.functional import normalize

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Loss, ParamGroup, ParamsT


@torch.no_grad()
def power_iteration(w: torch.Tensor, steps: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """Leading singular triplet (sigma, u, v) via bilateral power iteration (fp32/bf16)."""
    w = w.to(torch.bfloat16)
    v = torch.ones_like(w[..., :1, :].transpose(-2, -1))

    for _ in range(steps):
        v = normalize(w.transpose(-2, -1) @ (w @ v), dim=-2)

    u = normalize(w @ v, dim=-2)

    return u, v


@torch.no_grad()
def msign(x: torch.Tensor, steps: int) -> torch.Tensor:
    """Matrix sign via Newton-Schulz with Polar-Express coefficients."""
    transpose: bool = x.size(-2) > x.size(-1)

    x = x.mT if transpose else x
    x = normalize(x, p=2, dim=(-2, -1), eps=1e-7)

    x = x.to(torch.bfloat16)

    coefficients = [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ]

    for i in range(steps):
        coef_a, coef_b, coef_c = coefficients[i] if i < 8 else coefficients[-1]

        a = x @ x.mT
        b = torch.addmm(a, a, a, alpha=coef_c, beta=coef_b)
        x = torch.addmm(x, b, x, alpha=1.0, beta=coef_a)

    return x.mT if transpose else x


@torch.no_grad()
def compute_f_tensor(
    x: torch.Tensor,
    theta: torch.Tensor,
    lambda_value: Union[torch.Tensor, float],
    msign_steps: int = 8,
) -> torch.Tensor:
    """f(lambda) = <Θ, msign(G + lambdaΘ)>. Returns 0-d tensor (no GPU sync)."""
    z: torch.Tensor = x + lambda_value * theta
    phi: torch.Tensor = msign(z, steps=msign_steps)
    return (theta * phi).sum()


@torch.no_grad()
def find_bracket(
    x: torch.Tensor,
    theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    max_expansions: int = 10,
    msign_steps: int = 8,
    tolerance_f: float = 1e-8,
) -> Tuple[Optional[float], Optional[float], torch.Tensor, torch.Tensor]:
    """Find lambda_l < lambda_r such that: f(lambda_l) <= 0 <= f(lambda_r) with f monotone increasing.

    If f(initial_guess) is already near zero, returns a degenerate bracket.
    Otherwise, expands exponentially in the direction indicated by f0.
    """
    lambda_0 = initial_guess
    f0 = compute_f_tensor(x, theta, lambda_0, msign_steps)

    if abs(f0) < tolerance_f:
        return lambda_0, lambda_0, f0, f0

    step = initial_step if f0 < 0 else -initial_step

    lambda_prev = lambda_0
    f_prev = f0

    for _ in range(max_expansions):
        lambda_new = lambda_prev + step
        f_new = compute_f_tensor(x, theta, lambda_new, msign_steps)

        sign_prev = f_prev <= 0.0
        sign_new = f_new <= 0.0

        if sign_prev != sign_new:
            if f_prev <= 0 <= f_new:
                lambda_l, f_l = lambda_prev, f_prev
                lambda_r, f_r = lambda_new, f_new
            elif f_new <= 0 <= f_prev:
                lambda_l, f_l = lambda_new, f_new
                lambda_r, f_r = lambda_prev, f_prev
            elif abs(f_prev) <= abs(f_new):  # pragma: no cover
                lambda_l = lambda_r = lambda_prev
                f_l = f_r = f_prev
            else:  # pragma: no cover
                lambda_l = lambda_r = lambda_new
                f_l = f_r = f_new

            return lambda_l, lambda_r, f_l, f_r

        step *= 2.0
        lambda_prev, f_prev = lambda_new, f_new

    return None, None, f0, f0


@torch.no_grad()
def solve_lambda_with_bisection(
    x: torch.Tensor,
    theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    tolerance_f: float = 1e-6,
    max_iterations: int = 20,
    max_expansions: int = 10,
    msign_steps: int = 8,
) -> float:
    """Solve lambda such that f(lambda) = <Θ, msign(G + lambdaΘ)> = 0 using bisection.

    Assumes f is strictly monotone increasing.
    """
    lambda_l, lambda_r, f_l, f_r = find_bracket(
        x,
        theta,
        initial_guess=initial_guess,
        initial_step=initial_step,
        max_expansions=max_expansions,
        msign_steps=msign_steps,
        tolerance_f=tolerance_f,
    )
    if lambda_l is None or lambda_r is None:
        return 0.0

    if abs(f_l) < abs(f_r):
        best_lambda, best_f = lambda_l, f_l
    else:
        best_lambda, best_f = lambda_r, f_r

    if abs(best_f) <= tolerance_f:
        return best_lambda

    for _ in range(1, max_iterations + 1):
        lambda_mid = 0.5 * (lambda_l + lambda_r)

        f_mid: torch.Tensor = compute_f_tensor(x, theta, lambda_mid, msign_steps)

        if abs(f_mid) < abs(best_f):
            best_lambda, best_f = lambda_mid, f_mid

        if abs(f_mid) <= tolerance_f:
            return lambda_mid

        if f_mid < 0:
            lambda_l, f_l = lambda_mid, f_mid
        else:
            lambda_r, f_r = lambda_mid, f_mid

    return best_lambda


def compute_spectral_ball_update(
    weight: torch.Tensor,
    momentum: torch.Tensor,
    power_iteration_steps: int,
    msign_steps: int,
    solver_tolerance_f: float,
    solver_max_iterations: int,
) -> torch.Tensor:
    pass


class SpectralSphere(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.9,
        weight_decay: float = 1e-2,
        weight_decouple: bool = True,
        nesterov: bool = True,
        power_iteration_steps: int = 10,
        msign_steps: int = 5,
        solver_tolerance_f: float = 1e-8,
        solver_max_iterations: int = 100,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_range(momentum, 'momentum', 0.0, 1.0, range_type='[)')
        self.validate_positive(power_iteration_steps, 'power_iteration_steps')
        self.validate_positive(msign_steps, 'msign_steps')

        self.power_iteration_steps = power_iteration_steps
        self.msign_steps = msign_steps
        self.solver_tolerance_f = solver_tolerance_f
        self.solver_max_iterations = solver_max_iterations

        self.maximize = maximize

        defaults = {
            'lr': lr,
            'momentum': momentum,
            'nesterov': nesterov,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'SpectralSphere'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
