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


class RendererThread:
    def __init__(self, load_path: Path):
        device = torch.device("cuda:0")

        self.renderer = GaussianSplatting(device)
        self.renderer.init_gaussians(0, load_path)
        # self.renderer.init_axis()
        print(f"Number rendered gaussians: {self.renderer.num_points}")

        self.camera = Camera(400, 400, 480, 480, 200, 200, device)

        self.anglh = 0.0
        self.anglv = 0.0
        self.dis = 8.0
        self.size = 1.0

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.background: Tensor = torch.ones(3, device=device)
        self.other_background: Tensor = torch.zeros(3, device=device)

    def toggle_background(self):
        self.background, self.other_background = self.other_background, self.background

    def translate(self, x: float, y: float, z: float):
        self.x += x
        self.y += y
        self.z += z
        self.camera.translate(x, y, z)
        self._update_camera()

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

    def set_size(self, size: float):
        self.size = size

    def _update_camera(self):
        self.camera.orbit(self.x, self.y, self.z, self.dis, self.anglh, self.anglv)

    def render(self, width: int, height: int) -> Image:
        out_img, _, _ = self.renderer.render(self.camera, self.size, self.background)
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
        self.root.bind("<Key>", self.key_handler)

        self.panel = Label(self.root)
        self.panel.place(x=0, y=0)

        self.slider = Scale(self.root, from_=0, to=1, orient=HORIZONTAL)
        self.slider.bind("<Enter>", self._enter_no_drag)
        self.slider.bind("<Leave>", self._leave_no_drag)
        self.slider.set(1.0)
        self.slider.place(x=10, y=10)

        self.label = Label(self.root, text=f"{self.slider.get():.2f}")
        self.label.place(x=120, y=15)

        self.background_toggle = Button(self.root, text="Background", command=self._toggle_background)
        self.background_toggle.place(relx=1, x=-10, y=10, anchor=NE)

        self.prev_image: ImageTk.PhotoImage

        self.renderer = RendererThread(load_path)

        self.dragging = False
        self.drag_space = True
        self.last_x = 0.0
        self.last_y = 0.0

    def key_handler(self, event):
        match event.char:
            case 'w':
                self.renderer.translate(0, 0, -1)
            case 's':
                self.renderer.translate(0, 0, 1)
            case 'a':
                self.renderer.translate(-1, 0, 0)
            case 'd':
                self.renderer.translate(1, 0, 0)

        if event.keycode == 32:
            self.renderer.translate(0, 1, 0)
        if event.keycode == 16:
            self.renderer.translate(0, -1, 0)

    def run(self):
        self.render_loop()
        self.root.mainloop()

    def render_loop(self):
        size = self.slider.get()
        self.label.configure(text=f"{size:.2f}")

        self.renderer.set_size(size)
        img = self.renderer.render(800, 800)
        final_img = ImageTk.PhotoImage(img)
        self.panel.configure(image=final_img)
        self.prev_image = final_img
        self.root.after(20, self.render_loop)

    def _toggle_background(self):
        self.renderer.toggle_background()

    def _on_close(self):
        print("close")
        self.root.destroy()

    def _drag(self, event):
        if not self.dragging or not self.drag_space:
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
        self.drag_space = True

    def _scroll(self, event):
        self.renderer.add_distance(-float(event.delta)/240.0)

    def _enter_no_drag(self, event):
        self.drag_space = False

    def _leave_no_drag(self, event):
        self.last_x = event.x
        self.last_y = event.y
        if not self.dragging:
            self.drag_space = True


def main(
        load_path: Path
) -> None:
    viewer = Viewer(load_path)
    viewer.run()


if __name__ == '__main__':
    tyro.cli(main)
