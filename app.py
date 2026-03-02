from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from face_editor import FaceEditor


HAIR_COLOR_PRESETS = {
    "保持不变": 0,
    "金色": 18,
    "红棕": 8,
    "蓝黑": 105,
    "紫色": 145,
}


class FaceEditorApp:
    def __init__(self) -> None:
        self.editor = FaceEditor()
        self.original_bgr: np.ndarray | None = None
        self.edited_bgr: np.ndarray | None = None

        self.root = tk.Tk()
        self.root.title("CelebA 人脸编辑系统（桌面版）")
        self.root.geometry("1220x760")
        self.root.configure(bg="#f8fafc")

        self._build_style()
        self._build_layout()

    def _build_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Microsoft YaHei", 18, "bold"), background="#f8fafc")
        style.configure("Desc.TLabel", font=("Microsoft YaHei", 10), background="#f8fafc", foreground="#334155")
        style.configure("Panel.TFrame", background="#ffffff", borderwidth=1, relief="solid")
        style.configure("Panel.TLabel", background="#ffffff")

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=14, pady=(12, 4))
        ttk.Label(header, text="✨ CelebA 人脸编辑系统（非浏览器 App）", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="支持：发色调整 / 嘴巴大小 / 眼睛大小 / 瘦脸。上传图片后点击“应用编辑”。",
            style="Desc.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=14, pady=10)

        self.control_panel = ttk.Frame(body, style="Panel.TFrame", padding=12)
        self.control_panel.pack(side="left", fill="y")

        self.preview_panel = ttk.Frame(body, style="Panel.TFrame", padding=10)
        self.preview_panel.pack(side="right", fill="both", expand=True, padx=(12, 0))

        self._build_controls()
        self._build_preview()

    def _build_controls(self) -> None:
        ttk.Label(self.control_panel, text="参数设置", font=("Microsoft YaHei", 13, "bold"), style="Panel.TLabel").pack(anchor="w")

        ttk.Button(self.control_panel, text="上传人脸图片", command=self.load_image).pack(fill="x", pady=(12, 8))

        self.hair_color_var = tk.StringVar(value="保持不变")
        ttk.Label(self.control_panel, text="头发颜色", style="Panel.TLabel").pack(anchor="w", pady=(8, 2))
        hair_box = ttk.Combobox(
            self.control_panel,
            textvariable=self.hair_color_var,
            state="readonly",
            values=list(HAIR_COLOR_PRESETS.keys()),
        )
        hair_box.pack(fill="x")

        self.hair_sat_var = tk.DoubleVar(value=0.35)
        self._add_slider("头发饱和度增强", self.hair_sat_var, 0.0, 1.0)

        self.mouth_var = tk.DoubleVar(value=1.0)
        self._add_slider("嘴巴大小", self.mouth_var, 0.75, 1.35)

        self.eye_var = tk.DoubleVar(value=1.0)
        self._add_slider("眼睛大小", self.eye_var, 0.8, 1.4)

        self.slim_var = tk.DoubleVar(value=0.35)
        self._add_slider("瘦脸强度", self.slim_var, 0.0, 1.0)

        ttk.Button(self.control_panel, text="应用编辑", command=self.apply_edit).pack(fill="x", pady=(16, 6))
        ttk.Button(self.control_panel, text="保存结果", command=self.save_image).pack(fill="x", pady=6)
        ttk.Button(self.control_panel, text="重置参数", command=self.reset_params).pack(fill="x", pady=6)

        self.status_var = tk.StringVar(value="请先上传图片。")
        ttk.Label(
            self.control_panel,
            textvariable=self.status_var,
            wraplength=280,
            foreground="#0f766e",
            style="Panel.TLabel",
        ).pack(anchor="w", pady=(14, 0))

    def _add_slider(self, label: str, variable: tk.DoubleVar, start: float, end: float) -> None:
        ttk.Label(self.control_panel, text=label, style="Panel.TLabel").pack(anchor="w", pady=(10, 2))
        tk.Scale(
            self.control_panel,
            from_=start,
            to=end,
            resolution=0.01,
            orient="horizontal",
            variable=variable,
            bg="#ffffff",
            highlightthickness=0,
            length=280,
        ).pack(anchor="w")

    def _build_preview(self) -> None:
        top = ttk.Frame(self.preview_panel, style="Panel.TFrame")
        top.pack(fill="both", expand=True)

        self.input_label = ttk.Label(top, text="原图预览", style="Panel.TLabel")
        self.input_label.pack(side="left", fill="both", expand=True, padx=(6, 6))

        self.output_label = ttk.Label(top, text="编辑结果", style="Panel.TLabel")
        self.output_label.pack(side="right", fill="both", expand=True, padx=(6, 6))

    def _render_to_label(self, image_bgr: np.ndarray, target: ttk.Label) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        max_w, max_h = 430, 640
        scale = min(max_w / w, max_h / h, 1.0)
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)
        target.configure(image=photo, text="")
        target.image = photo

    def load_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择人脸图片",
            filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )
        if not file_path:
            return

        bgr = cv2.imread(file_path)
        if bgr is None:
            messagebox.showerror("错误", "无法读取图片，请重新选择。")
            return

        self.original_bgr = bgr
        self.edited_bgr = bgr.copy()
        self._render_to_label(self.original_bgr, self.input_label)
        self._render_to_label(self.edited_bgr, self.output_label)
        self.status_var.set(f"已加载图片：{Path(file_path).name}")

    def apply_edit(self) -> None:
        if self.original_bgr is None:
            self.status_var.set("请先上传图片。")
            return

        edited, msg = self.editor.edit(
            self.original_bgr,
            hair_hue_shift=HAIR_COLOR_PRESETS.get(self.hair_color_var.get(), 0),
            hair_saturation_boost=float(self.hair_sat_var.get()),
            mouth_scale=float(self.mouth_var.get()),
            eye_scale=float(self.eye_var.get()),
            slim_face_strength=float(self.slim_var.get()),
        )

        self.edited_bgr = edited
        self._render_to_label(self.edited_bgr, self.output_label)
        self.status_var.set(msg)

    def save_image(self) -> None:
        if self.edited_bgr is None:
            self.status_var.set("没有可保存的结果，请先编辑图片。")
            return

        out_path = filedialog.asksaveasfilename(
            title="保存编辑结果",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if not out_path:
            return

        cv2.imwrite(out_path, self.edited_bgr)
        self.status_var.set(f"已保存到：{out_path}")

    def reset_params(self) -> None:
        self.hair_color_var.set("保持不变")
        self.hair_sat_var.set(0.35)
        self.mouth_var.set(1.0)
        self.eye_var.set(1.0)
        self.slim_var.set(0.35)
        self.status_var.set("参数已重置。")

    def run(self) -> None:
        self.root.mainloop()

    clear_button.click(
        fn=lambda: (None, None, "已清空"),
        inputs=[],
        outputs=[input_image, output_image, status],
    )


if __name__ == "__main__":
    app = FaceEditorApp()
    app.run()
