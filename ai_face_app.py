from __future__ import annotations

import base64
import io
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import requests
from PIL import Image, ImageTk


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _base64_to_image(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


class WebUIClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/sdapi/v1/options", timeout=5)
            return r.ok
        except requests.RequestException:
            return False

    def img2img_with_controlnet(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        denoise_strength: float,
        cfg_scale: float,
        steps: int,
        sampler_name: str,
        controlnet_model: str = "control_v11f1p_sd15_depth [cfd03158]",
        control_weight: float = 1.0,
    ) -> Image.Image:
        img_b64 = _image_to_base64(image)
        payload: dict[str, Any] = {
            "init_images": [img_b64],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "denoising_strength": denoise_strength,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "sampler_name": sampler_name,
            "resize_mode": 0,
            "alwayson_scripts": {
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "module": "depth",
                            "model": controlnet_model,
                            "weight": control_weight,
                            "resize_mode": "Crop and Resize",
                            "processor_res": 512,
                            "threshold_a": 0,
                            "threshold_b": 0,
                            "guidance": 1.0,
                            "guidance_start": 0.0,
                            "guidance_end": 1.0,
                            "control_mode": 0,
                            "pixel_perfect": True,
                            "image": {"image": img_b64, "mask": None},
                        }
                    ]
                }
            },
        }

        resp = requests.post(
            f"{self.base_url}/sdapi/v1/img2img",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("images"):
            raise RuntimeError("WebUI 返回为空，请检查模型/ControlNet 配置")
        return _base64_to_image(data["images"][0])


class AIFaceApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("AI 人脸精修系统（Stable Diffusion WebUI API）")
        self.root.geometry("1300x780")

        self.client_url_var = tk.StringVar(value="http://127.0.0.1:7860")
        self.image_path: Path | None = None
        self.input_image: Image.Image | None = None
        self.output_image: Image.Image | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = self.root
        style = ttk.Style()
        style.theme_use("clam")

        top = ttk.Frame(root, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text="WebUI 地址:").pack(side="left")
        ttk.Entry(top, textvariable=self.client_url_var, width=40).pack(side="left", padx=(6, 8))
        ttk.Button(top, text="检查连接", command=self.check_connection).pack(side="left", padx=(0, 12))
        ttk.Label(top, text="提示：先用 --api --listen 启动 WebUI，且安装 ControlNet").pack(side="left")

        body = ttk.Frame(root, padding=(12, 6, 12, 12))
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, padding=10)
        left.pack(side="left", fill="y")

        ttk.Button(left, text="上传图片", command=self.load_image).pack(fill="x", pady=4)

        self.eye_var = tk.DoubleVar(value=1.1)
        self.lip_var = tk.DoubleVar(value=1.08)
        self.young_var = tk.DoubleVar(value=0.75)
        self.skin_var = tk.DoubleVar(value=0.70)
        self.nose_var = tk.DoubleVar(value=0.95)

        self._slider(left, "眼睛放大", self.eye_var, 0.85, 1.35)
        self._slider(left, "嘴唇丰盈", self.lip_var, 0.85, 1.35)
        self._slider(left, "年轻化", self.young_var, 0.0, 1.0)
        self._slider(left, "皮肤质感", self.skin_var, 0.0, 1.0)
        self._slider(left, "鼻翼精修", self.nose_var, 0.85, 1.15)

        self.denoise_var = tk.DoubleVar(value=0.35)
        self.cfg_var = tk.DoubleVar(value=7.0)
        self.steps_var = tk.IntVar(value=28)
        self.control_weight_var = tk.DoubleVar(value=1.0)
        self.sampler_var = tk.StringVar(value="DPM++ 2M Karras")

        self._slider(left, "重绘强度 denoise", self.denoise_var, 0.2, 0.55)
        self._slider(left, "CFG", self.cfg_var, 4.0, 10.0)
        self._slider(left, "ControlNet 权重", self.control_weight_var, 0.5, 1.6)
        self._slider(left, "步数", self.steps_var, 18, 45, resolution=1)

        ttk.Label(left, text="采样器").pack(anchor="w", pady=(8, 2))
        ttk.Combobox(
            left,
            textvariable=self.sampler_var,
            values=["Euler a", "Euler", "DPM++ 2M Karras", "DPM++ SDE Karras", "UniPC"],
            state="readonly",
        ).pack(fill="x")

        ttk.Button(left, text="应用 AI 精修", command=self.run_edit).pack(fill="x", pady=(12, 4))
        ttk.Button(left, text="保存结果", command=self.save_image).pack(fill="x", pady=4)

        self.status_var = tk.StringVar(value="请先上传图片。")
        ttk.Label(left, textvariable=self.status_var, wraplength=300, foreground="#0f766e").pack(anchor="w", pady=(12, 0))

        right = ttk.Frame(body)
        right.pack(side="right", fill="both", expand=True)
        self.input_label = ttk.Label(right, text="原图")
        self.input_label.pack(side="left", fill="both", expand=True, padx=6)
        self.output_label = ttk.Label(right, text="AI 结果")
        self.output_label.pack(side="right", fill="both", expand=True, padx=6)

    def _slider(self, parent: ttk.Frame, title: str, var: tk.Variable, start: float, end: float, resolution: float = 0.01) -> None:
        ttk.Label(parent, text=title).pack(anchor="w", pady=(8, 2))
        tk.Scale(parent, from_=start, to=end, resolution=resolution, orient="horizontal", variable=var, length=300).pack(anchor="w")

    def _render(self, image: Image.Image, target: ttk.Label) -> None:
        max_w, max_h = 460, 680
        w, h = image.size
        s = min(max_w / w, max_h / h, 1.0)
        view = image.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
        tkimg = ImageTk.PhotoImage(view)
        target.configure(image=tkimg, text="")
        target.image = tkimg

    def check_connection(self) -> None:
        c = WebUIClient(self.client_url_var.get())
        if c.health():
            self.status_var.set("WebUI API 连接成功。")
        else:
            self.status_var.set("连接失败：请确认 WebUI 已用 --api --listen 启动。")

    def load_image(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if not p:
            return
        img = Image.open(p).convert("RGB")
        self.image_path = Path(p)
        self.input_image = img
        self.output_image = img.copy()
        self._render(self.input_image, self.input_label)
        self._render(self.output_image, self.output_label)
        self.status_var.set(f"已加载：{self.image_path.name}")

    def _build_prompt(self) -> str:
        eye = self.eye_var.get()
        lip = self.lip_var.get()
        young = self.young_var.get()
        skin = self.skin_var.get()
        nose = self.nose_var.get()

        prompt = [
            "ultra detailed RAW portrait, realistic skin, high fidelity facial details",
            "keep same identity, keep same pose, keep same face layout",
        ]
        if eye > 1.02:
            prompt.append("slightly larger natural eyes")
        if lip > 1.02:
            prompt.append("slightly fuller lips")
        if young > 0.05:
            prompt.append(f"younger face by {young:.2f}")
        if skin > 0.05:
            prompt.append(f"clean smooth skin by {skin:.2f}")
        if nose < 0.99:
            prompt.append("slightly narrower nose wings")

        return ", ".join(prompt)

    def run_edit(self) -> None:
        if self.input_image is None:
            self.status_var.set("请先上传图片。")
            return

        negative = "deformed face, distorted eyes, extra face, lowres, blurry, cartoon, painting"
        prompt = self._build_prompt()

        try:
            client = WebUIClient(self.client_url_var.get())
            out = client.img2img_with_controlnet(
                image=self.input_image,
                prompt=prompt,
                negative_prompt=negative,
                denoise_strength=float(self.denoise_var.get()),
                cfg_scale=float(self.cfg_var.get()),
                steps=int(self.steps_var.get()),
                sampler_name=self.sampler_var.get(),
                control_weight=float(self.control_weight_var.get()),
            )
            self.output_image = out
            self._render(out, self.output_label)
            self.status_var.set("AI 精修完成。")
        except Exception as e:
            messagebox.showerror("推理失败", str(e))
            self.status_var.set("推理失败，请检查 WebUI/ControlNet/模型是否正确安装。")

    def save_image(self) -> None:
        if self.output_image is None:
            self.status_var.set("没有可保存图片。")
            return
        out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not out:
            return
        self.output_image.save(out)
        self.status_var.set(f"已保存：{out}")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    app = AIFaceApp()
    app.run()
