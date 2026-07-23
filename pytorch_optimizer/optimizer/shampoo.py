import torch

from pytorch_optimizer.base.exception import NoComplexParameterError, NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.shampoo_utils import (
    LayerWiseGrafting,
    PreConditioner,
    PreConditionerType,
    build_graft,
    compute_power_svd,
)


class Shampoo(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        weight_decouple: bool = False,
        fixed_decay: bool = False,
        preconditioning_compute_steps: int = 1,
        matrix_eps: float = 1e-6,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_range(momentum, 'momentum', 0.0, 1.0)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_step(preconditioning_compute_steps, 'preconditioning_compute_steps')
        self.validate_non_negative(matrix_eps, 'matrix_eps')

        self.preconditioning_compute_steps = preconditioning_compute_steps
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'matrix_eps': matrix_eps,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Shampoo'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass


class ScalableShampoo(BaseOptimizer):

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        moving_average_for_momentum: bool = False,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
        decoupled_learning_rate: bool = True,
        inverse_exponent_override: int = 0,
        start_preconditioning_step: int = 25,
        preconditioning_compute_steps: int = 1000,
        statistics_compute_steps: int = 1,
        block_size: int = 512,
        skip_preconditioning_rank_lt: int = 1,
        no_preconditioning_for_layers_with_dim_gt: int = 8192,
        shape_interpretation: bool = True,
        graft_type: int = LayerWiseGrafting.SGD,
        pre_conditioner_type: int = PreConditionerType.ALL,
        nesterov: bool = True,
        diagonal_eps: float = 1e-10,
        matrix_eps: float = 1e-6,
        use_svd: bool = False,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_step(start_preconditioning_step, 'start_preconditioning_step')
        self.validate_step(preconditioning_compute_steps, 'preconditioning_compute_steps')
        self.validate_step(statistics_compute_steps, 'statistics_compute_steps')
        self.validate_non_negative(diagonal_eps, 'diagonal_eps')
        self.validate_non_negative(matrix_eps, 'matrix_eps')

        self.inverse_exponent_override = inverse_exponent_override
        self.start_preconditioning_step = start_preconditioning_step
        self.preconditioning_compute_steps = preconditioning_compute_steps
        self.statistics_compute_steps = statistics_compute_steps
        self.block_size = block_size
        self.skip_preconditioning_rank_lt = skip_preconditioning_rank_lt
        self.no_preconditioning_for_layers_with_dim_gt = no_preconditioning_for_layers_with_dim_gt
        self.shape_interpretation = shape_interpretation
        self.graft_type = graft_type
        self.pre_conditioner_type = pre_conditioner_type
        self.diagonal_eps = diagonal_eps
        self.matrix_eps = matrix_eps
        self.use_svd = use_svd
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'decoupled_weight_decay': decoupled_weight_decay,
            'decoupled_learning_rate': decoupled_learning_rate,
            'moving_average_for_momentum': moving_average_for_momentum,
            'nesterov': nesterov,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'ScalableShampoo'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def is_precondition_step(self, step: int) -> bool:
        pass

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
