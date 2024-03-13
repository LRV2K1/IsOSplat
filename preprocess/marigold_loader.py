from pathlib import Path
import numpy as np


def load_depth_map(depth_path: Path) -> np.ndarray:
    depth_map = np.load(depth_path)
    return depth_map