from pathlib import Path
from PIL import Image

import torch
from torch import Tensor


def image_path_to_tensor(image_path: Path, device: torch.device) -> tuple[Tensor, Tensor]:
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_transform = transform(img)
    img_tensor = img_transform.permute(1, 2, 0)[..., :3]
    if img_transform.shape[0] > 3:
        img_alpha_tensor = img_transform.permute(1, 2, 0)[..., 3]
    else:
        img_alpha_tensor = torch.ones(img_tensor.shape[0], img_tensor.shape[0]) * 1.0
    img_tensor = img_tensor.to(device=device)
    img_alpha_tensor = img_alpha_tensor.to(device=device)
    return img_tensor, img_alpha_tensor
