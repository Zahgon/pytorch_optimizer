from collections import defaultdict
from typing import Callable, Dict

import torch
from torch.optim import Optimizer

from pytorch_optimizer.base.optimizer import BaseOptimizer
from pytorch_optimizer.base.type import Closure, Defaults, Loss, OptimizerInstanceOrClass, ParamGroup, State


class Lookahead(BaseOptimizer):

    def __init__(
        self,
        optimizer: OptimizerInstanceOrClass,
        k: int = 5,
        alpha: float = 0.5,
        pullback_momentum: str = 'none',
        **kwargs,
    ) -> None:
        self.validate_positive(k, 'k')
        self.validate_range(alpha, 'alpha', 0.0, 1.0)
        self.validate_options(pullback_momentum, 'pullback_momentum', ['none', 'reset', 'pullback'])

        self.optimizer: Optimizer = self.load_optimizer(optimizer, **kwargs)

        self._optimizer_step_pre_hooks: Dict[int, Callable] = {}
        self._optimizer_step_post_hooks: Dict[int, Callable] = {}

        self.alpha = alpha
        self.k = k
        self.pullback_momentum = pullback_momentum

        self.state: State = defaultdict(dict)

        for group in self.param_groups:
            if 'counter' not in group:
                group['counter'] = 0

            for p in group['params']:
                state = self.state[p]
                state['slow_params'] = torch.empty_like(p)
                state['slow_params'].copy_(p)
                if self.pullback_momentum == 'pullback':
                    state['slow_momentum'] = torch.zeros_like(p)

        self.defaults: Defaults = {
            'lookahead_alpha': alpha,
            'lookahead_k': k,
            'lookahead_pullback_momentum': pullback_momentum,
            **self.optimizer.defaults,
        }

    @property
    def param_groups(self):
        pass

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'k': self.k,
            'pullback_momentum': self.pullback_momentum,
        }

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    def init_group(self, group: ParamGroup, **kwargs) -> None:
        pass

    def backup_and_load_cache(self) -> None:
        pass

    def clear_and_load_backup(self) -> None:
        pass

    def state_dict(self) -> State:
        pass

    def load_state_dict(self, state: State) -> None:
        pass

    @torch.no_grad()
    def update(self, group: Dict):
        for p in group['params']:
            if p.grad is None:
                continue

            state = self.state[p]

            slow = state['slow_params']

            p.mul_(self.alpha).add_(slow, alpha=1.0 - self.alpha)
            slow.copy_(p)

            if 'momentum_buffer' not in self.optimizer.state[p]:
                self.optimizer.state[p]['momentum_buffer'] = torch.zeros_like(p)

            if self.pullback_momentum == 'pullback':
                internal_momentum = self.optimizer.state[p]['momentum_buffer']
                self.optimizer.state[p]['momentum_buffer'] = internal_momentum.mul_(self.alpha).add_(
                    state['slow_momentum'], alpha=1.0 - self.alpha
                )
                state['slow_momentum'] = self.optimizer.state[p]['momentum_buffer']
            elif self.pullback_momentum == 'reset':
                self.optimizer.state[p]['momentum_buffer'] = torch.zeros_like(p)

    def step(self, closure: Closure = None) -> Loss:
        pass
