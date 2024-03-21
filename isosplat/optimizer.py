import time
from typing import Optional

import torch
from torch import Tensor, optim, nn

from .utils import get_expon_lr_func


class Optimizer:
    def __init__(self):
        self.optimizer: Optional[optim.Adam] = None

    def load_tensor_dict(self, tensor_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        optimizer_params = []
        optimizable_tensors = {}
        for key in tensor_dict:
            tensor, lr = tensor_dict[key]
            tensor.requires_grad = True
            optimizer_params.append(
                {'params': tensor, 'lr': lr, 'name': key}
            )
            optimizable_tensors[key] = tensor

        self.optimizer = optim.Adam(optimizer_params, lr=0.0, eps=1e-15)
        return optimizable_tensors

    def back_propagate_loss(self, loss: Tensor) -> float:
        self.optimizer.zero_grad()
        start = time.time()

        loss.backward()
        torch.cuda.synchronize()

        t2 = time.time() - start

        self.optimizer.step()
        return t2
    
    def set_learning_rate_scheduler(self, init_lr: float, final_lr: float, lr_delay_mult: float, lr_max_steps: int):
        self.mean_lr_sheduler = get_expon_lr_func(lr_init=init_lr,
                                                    lr_final=final_lr,
                                                    lr_delay_mult=lr_delay_mult,
                                                    max_steps=lr_max_steps)

    def update_learning_rate(self, itr: int) -> float:
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "means":
                lr = self.mean_lr_sheduler(itr)
                param_group['lr'] = lr
                return lr
            
    def get_learning_rate(self):
        return self.lr

    def prune_optimizer(self, mask: Tensor) -> dict[str, Tensor]:
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def replace_optimizer_tensor(self, new_tensor: Tensor, name: str) -> dict[str, Tensor]:
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(new_tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(new_tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def replace_optimizer_tensors(self, new_tensors: dict[str, Tensor]) -> dict[str, Tensor]:
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            if not (name in new_tensors):
                continue
            new_tensor = new_tensors[name]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            stored_state["exp_avg"] = torch.zeros_like(new_tensor)
            stored_state["exp_avg_sq"] = torch.zeros_like(new_tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_optimizer_tensors(self, add_tensors: dict[str, Tensor]) -> dict[str, Tensor]:
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = add_tensors[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
