from typing import List, Optional, Union

import torch

from pytorch_optimizer.base.exception import NoSparseGradientError
from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Betas, Closure, Defaults, Loss, ParamGroup, ParamsT
from pytorch_optimizer.optimizer.utils import get_global_gradient_norm


class Lamb(BaseOptimizer):

    clamp: float = 10.0

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Betas = (0.9, 0.999),
        weight_decay: float = 0.0,
        weight_decouple: bool = True,
        fixed_decay: bool = False,
        rectify: bool = False,
        degenerated_to_sgd: bool = False,
        n_sma_threshold: int = 5,
        grad_averaging: bool = True,
        max_grad_norm: float = 1.0,
        adam: bool = False,
        pre_norm: bool = False,
        eps: float = 1e-6,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        **kwargs,
    ):
        self.validate_learning_rate(lr)
        self.validate_betas(betas)
        self.validate_non_negative(weight_decay, 'weight_decay')
        self.validate_non_negative(max_grad_norm, 'max_grad_norm')
        self.validate_non_negative(eps, 'eps')

        self.degenerated_to_sgd = degenerated_to_sgd
        self.n_sma_threshold = n_sma_threshold
        self.pre_norm = pre_norm
        self.foreach = foreach
        self.maximize = maximize

        defaults: Defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'weight_decouple': weight_decouple,
            'fixed_decay': fixed_decay,
            'rectify': rectify,
            'grad_averaging': grad_averaging,
            'max_grad_norm': max_grad_norm,
            'adam': adam,
            'eps': eps,
            'foreach': foreach,
            **kwargs,
        }

        super().__init__(params, defaults)

    def __str__(self) -> str:
        return 'Lamb'

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def _can_use_foreach(self, group: ParamGroup) -> bool:
        pass

    def _step_foreach(
        self,
        group: ParamGroup,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        grad_norm: Union[torch.Tensor, float],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        step_size: float,
    ) -> None:
        pass

    @torch.no_grad()
    def get_global_gradient_norm(self) -> Union[torch.Tensor, float]:
        if self.defaults['max_grad_norm'] == 0.0:
            return 1.0

        global_grad_norm = get_global_gradient_norm(self.param_groups)
        global_grad_norm.sqrt_().add_(self.defaults['eps'])

        return torch.clamp(self.defaults['max_grad_norm'] / global_grad_norm, max=1.0)

    def update(
        self,
        p: torch.Tensor,
        group: ParamGroup,
        grad_norm: Union[torch.Tensor, float],
        n_sma: float,
        step_size: float,
        beta1: float,
        beta2: float,
        beta3: float,
    ) -> None:
        grad = p.grad
        if grad is None:
            return

        if self.pre_norm:
            grad.div_(grad_norm)

        self.maximize_gradient(grad, maximize=self.maximize)

        state = self.state[p]

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

        p, grad, exp_avg, exp_avg_sq = self.view_as_real(p, grad, exp_avg, exp_avg_sq)

        s_grad = self.get_adanorm_gradient(
            grad=grad,
            adanorm=group.get('adanorm', False),
            exp_grad_norm=state.get('exp_grad_adanorm', None),
            r=group.get('adanorm_r', None),
        )

        exp_avg.mul_(beta1).add_(s_grad, alpha=beta3)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        self.apply_weight_decay(
            p=p,
            grad=None,
            lr=group['lr'],
            weight_decay=group['weight_decay'],
            weight_decouple=group['weight_decouple'],
            fixed_decay=group['fixed_decay'],
        )

        de_nom: Optional[torch.Tensor] = None

        if group['rectify']:
            update = p.clone()
            if n_sma >= self.n_sma_threshold:
                de_nom = exp_avg_sq.sqrt().add_(group['eps'])
                update.addcdiv_(exp_avg, de_nom, value=-step_size)
            else:
                update.add_(exp_avg, alpha=-step_size)
        else:
            update = exp_avg / exp_avg_sq.sqrt().add_(group['eps'])

        weight_norm = torch.linalg.norm(p).clamp_(min=0, max=self.clamp)
        p_norm = torch.linalg.norm(update)
        trust_ratio: float = 1.0 if weight_norm == 0 or p_norm == 0 else weight_norm / (p_norm + group['eps'])

        state['weight_norm'] = weight_norm
        state['adam_norm'] = p_norm
        state['trust_ratio'] = trust_ratio

        if group['adam']:
            trust_ratio = 1.0

        if group['rectify']:
            if n_sma >= self.n_sma_threshold:
                p.addcdiv_(exp_avg, de_nom, value=-step_size * trust_ratio)
            else:
                p.add_(exp_avg, alpha=-step_size * trust_ratio)
        else:
            p.add_(update, alpha=-step_size * trust_ratio)

    @torch.no_grad()
    def step(self, closure: Closure = None) -> Loss:
        pass
