import torch
import tyro
from pathlib import Path
from isosplat.gaussian_splatting import GaussianSplatting
from isosplat.camera import Camera

from tkinter import *
from tkinter.ttk import *

from torch import Tensor

from PIL import Image, ImageTk
import numpy as np

import threading


class RendererThread:
    def __init__(self, load_path: Path, panel: Label):
        device = torch.device("cuda:0")

        self.renderer = GaussianSplatting(device)
        self.renderer.init_gaussians(0, load_path)

        self.camera = Camera(400, 400, 200, 200, 0, 10, device)

        self.thread = threading.Thread(target=self._run)
        self.running = False

        self.panel = panel
        self.prev_image: ImageTk.PhotoImage

        self.anglh = 0.0
        self.anglv = 0.0
        self.dis = 8.0

        self.background: Tensor = torch.ones(3, device=device)

    def add_angle(self, anglh: float, anglv: float):
        self.anglh += anglh
        self.anglv += anglv
        self._update_camera()

    def add_distance(self, dis: float):
        self.dis += dis
        self._update_camera()

    def _update_camera(self):
        self.camera.orbit(0.0, 0.0, 0.0, self.dis, self.anglh, self.anglv)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _run(self):
        while self.running:
            out_img, _, _ = self.renderer.render(self.camera, self.background)
            img = Image.fromarray((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
            img = img.resize((800, 800))
            img = ImageTk.PhotoImage(img)
            try:
                self.panel.configure(image=img)
                self.prev_image = img
            except Exception as e:
                print("display failure:   ", e)


class Viewer:
    def __init__(self, load_path: Path):
        self.root = Tk()
        self.root.title("IsOSplat")
        self.root.geometry("800x800")

        self.root.bind("<B1-Motion>", self._drag)
        self.root.bind("<Button-1>", self._clicked)
        self.root.bind("<ButtonRelease-1>", self._released)
        self.root.bind("<MouseWheel>", self._scroll)

        self.panel = Label(self.root)
        self.panel.place(x=0, y=0)

        self.renderer = RendererThread(load_path, self.panel)
        self.renderer.start()

        self.dragging = False
        self.last_x = 0.0
        self.last_y = 0.0

    def run(self):
        self.root.mainloop()
        self.renderer.stop()

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

    # root = Tk()
    # root.title("IsOSplat")
    # root.geometry("800x800")
    #
    # panel = Label(root)
    # panel.place(x=0, y=0)
    #
    # renderer = RendererThread(load_path, panel)
    # renderer.start()
    #
    # # label = Label(root, text="Hello World !")
    # # # label.pack()
    # # label.place(x=0, y=0)
    # #
    # # frame = Frame(root)
    # # frame.place(x=0, y=0)
    # #
    # # button = Button(root, text="button", command=clicked)
    # # # button.pack()
    # # button.place(x=10,y=10)
    #
    # # root.bind("<Button-1>", lambda x: print("Left click"))
    # # root.bind("<Button>", lambda x: print("Right click"))
    # # root.bind("<ButtonRelease-1>", lambda x: print("Left release"))
    # # root.bind("<ButtonRelease>", lambda x: print("Right release"))
    #
    # # root.bind("<B1-Motion>", drag)
    #
    # root.mainloop()
    # renderer.stop()


def clicked():
    print("click")


if __name__ == '__main__':
    tyro.cli(main)
