import time
from typing import Optional

import torch
from torch import Tensor, optim, nn


class Optimizer:
    def __init__(self, l_ssim: float = 0.2, l_depth: float = 0.1, l_smooth: float = 0.1):
        self.optimizer: Optional[optim.Adam] = None

        self.lr = 0.1
        self.l_ssim = l_ssim
        self.l_depth = l_depth
        self.l_smooth = l_smooth

    def load_tensor_dict(self, tensor_dict: dict[str, Tensor], lr: float = 0.01) -> dict[str, Tensor]:
        optimizer_params = []
        optimizable_tensors = {}
        for key in tensor_dict:
            tensor = tensor_dict[key]
            tensor.requires_grad = True
            optimizer_params.append(
                {'params': tensor, 'name': key}
            )
            optimizable_tensors[key] = tensor

        self.lr = lr
        self.optimizer = optim.Adam(optimizer_params, lr)
        return optimizable_tensors

    def back_propagate_loss(self, loss: Tensor) -> float:
        self.optimizer.zero_grad()
        start = time.time()

        loss.backward()
        torch.cuda.synchronize()

        t2 = time.time() - start

        self.optimizer.step()
        return t2

    def update_learning_rate(self, lr: float, name: str):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == name:
                param_group['lr'] = lr
                self.lr = lr
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
