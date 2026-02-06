# adopt_minimal.py
import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List, Optional, Tuple, Union, Callable, Iterable

__all__ = ["ADOPT"]


class ADOPT(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.9999),
        eps: float = 1e-6,
        clip_lambda: Optional[Callable[[int], float]] = lambda step: step**0.25,
        weight_decay: float = 0.0,
        decouple: bool = False,
    ):
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.clip_lambda = clip_lambda
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decouple=decouple,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("decouple", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            decouple = group["decouple"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("ADOPT does not support sparse gradients")

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step = state["step"] + 1

                # Decoupled weight decay
                if weight_decay != 0 and decouple:
                    p.mul_(1 - lr * weight_decay)

                # L2 weight decay (original)
                if weight_decay != 0 and not decouple:
                    grad = grad.add(p, alpha=weight_decay)

                # First step: initialize exp_avg_sq
                if step == 1:
                    exp_avg_sq.addcmul_(grad, grad.conj())
                    state["step"] = step
                    continue

                # Compute normalized gradient
                denom = exp_avg_sq.sqrt().clamp_min(eps)
                normed_grad = grad.div(denom)

                # Gradient clipping
                if self.clip_lambda is not None:
                    clip_val = self.clip_lambda(step)
                    normed_grad.clamp_(-clip_val, clip_val)

                # Update exp_avg
                exp_avg.lerp_(normed_grad, 1 - beta1)

                # Update parameters
                p.add_(exp_avg, alpha=-lr)

                # Update exp_avg_sq
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                state["step"] = step

        return loss