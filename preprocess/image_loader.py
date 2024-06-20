from pathlib import Path
from PIL import Image
import numpy as np
import os

import torch
from torch import Tensor


def image_path_to_tensor(image_path: Path, device: torch.device, image_scale = 1.0) -> tuple[Tensor, Tensor]:
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    width = int(img.width * image_scale)
    height = int(img.height * image_scale)
    img = img.resize((width, height))

    transform = transforms.ToTensor()
    img_transform = transform(img)
    img_tensor = img_transform.permute(1, 2, 0)[..., :3]
    if img_transform.shape[0] > 3:
        img_alpha_tensor = img_transform.permute(1, 2, 0)[..., 3]
    else:
        img_alpha_tensor = torch.ones(img_tensor.shape[0], img_tensor.shape[1]) * 1.0
    img_tensor = img_tensor.to(device=device)
    img_alpha_tensor = img_alpha_tensor.to(device=device)
    return img_tensor, img_alpha_tensor


def save_img_from_tensor(img: Tensor, image_path: Path, name: str):
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    image = Image.fromarray((img.detach().cpu().numpy() * 255).astype(np.uint8))
    image.save(f"{image_path}/{name}.png")
