import torch
import tyro
from pathlib import Path
from isosplat.gaussian_splatting import GaussianSplatting
from isosplat.camera import Camera
import math

from tkinter import *
from tkinter.ttk import *

from torch import Tensor

from PIL import Image, ImageTk
import numpy as np

import threading


class RendererThread:
    def __init__(self, load_path: Path):
        device = torch.device("cuda:0")

        self.renderer = GaussianSplatting(device)
        self.renderer.init_gaussians(0, load_path)
        print(f"Number rendered gaussians: {self.renderer.num_points}")

        self.camera = Camera(400, 400, 200, 200, 0, 10, device)

        self.anglh = 0.0
        self.anglv = 0.0
        self.dis = 8.0

        self.background: Tensor = torch.ones(3, device=device)

    def add_angle(self, anglh: float, anglv: float):
        self.anglh += anglh
        self.anglh = self.anglh % (2 * math.pi)
        self.anglv += anglv
        self.anglv = min(self.anglv, 0.5*math.pi)
        self.anglv = max(self.anglv, -0.5*math.pi)
        self._update_camera()

    def add_distance(self, dis: float):
        self.dis += dis
        self.dis = max(0.1, self.dis)
        self._update_camera()

    def _update_camera(self):
        self.camera.orbit(0.0, 0.0, 0.0, self.dis, self.anglh, self.anglv)

    def render(self, width: int, height: int) -> Image:
        out_img, _, _ = self.renderer.render(self.camera, self.background)
        img = Image.fromarray((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        return img.resize((width, height))


class Viewer:
    def __init__(self, load_path: Path):
        self.root = Tk()
        self.root.title("IsOSplat")
        self.root.geometry("800x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.root.bind("<B1-Motion>", self._drag)
        self.root.bind("<Button-1>", self._clicked)
        self.root.bind("<ButtonRelease-1>", self._released)
        self.root.bind("<MouseWheel>", self._scroll)

        self.panel = Label(self.root)
        self.panel.place(x=0, y=0)

        self.prev_image: ImageTk.PhotoImage

        self.renderer = RendererThread(load_path)

        self.dragging = False
        self.last_x = 0.0
        self.last_y = 0.0

    def run(self):
        # self.renderer.start()
        self.render_loop()
        self.root.mainloop()

    def render_loop(self):
        img = self.renderer.render(800, 800)
        final_img = ImageTk.PhotoImage(img)
        self.panel.configure(image=final_img)
        self.prev_image = final_img
        self.root.after(20, self.render_loop)

    def _on_close(self):
        print("close")
        self.root.destroy()

    def _drag(self, event):
        if not self.dragging:
            return

        dx = float(self.last_x - event.x)/25.0
        dy = float(event.y - self.last_y)/25.0
        self.renderer.add_angle(dx, dy)

        self.last_x = event.x
        self.last_y = event.y

    def _clicked(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self.dragging = True

    def _released(self, event):
        self.dragging = False

    def _scroll(self, event):
        self.renderer.add_distance(-float(event.delta)/240.0)


def main(
        load_path: Path
) -> None:
    viewer = Viewer(load_path)
    viewer.run()


if __name__ == '__main__':
    tyro.cli(main)
