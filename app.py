from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from face_editor import FaceEditor

editor = FaceEditor()


def process(
    image: np.ndarray,
    hair_color: str,
    hair_sat_boost: float,
    mouth_size: float,
    slim_face: float,
):
    if image is None:
        return None, "请先上传图片。"

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    hue_map = {
        "保持不变": 0,
        "金色": 18,
        "红棕": 8,
        "蓝黑": 105,
        "紫色": 145,
    }

    out, msg = editor.edit(
        bgr,
        hair_hue_shift=hue_map.get(hair_color, 0),
        hair_saturation_boost=hair_sat_boost,
        mouth_scale=mouth_size,
        slim_face_strength=slim_face,
    )
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return rgb, msg


with gr.Blocks(theme=gr.themes.Soft(primary_hue="rose"), title="CelebA 人脸编辑系统") as demo:
    gr.Markdown(
        """
# ✨ CelebA 人脸编辑系统
上传一张人脸图片（推荐 CelebA 对齐图像或清晰正脸），即可实时尝试以下美颜编辑：
- 头发颜色调整（多种预设）
- 嘴巴大小调整
- 瘦脸强度调整

> 提示：这是轻量本地版本，便于快速原型验证；后续可替换为基于 CelebA 训练的 GAN/扩散模型以获得更强可控性。
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传人脸", type="numpy")
            hair_color = gr.Dropdown(
                label="头发颜色", choices=["保持不变", "金色", "红棕", "蓝黑", "紫色"], value="保持不变"
            )
            hair_sat = gr.Slider(0, 1.0, value=0.35, step=0.05, label="头发饱和度增强")
            mouth_size = gr.Slider(0.75, 1.35, value=1.0, step=0.01, label="嘴巴大小")
            slim_face = gr.Slider(0, 1.0, value=0.35, step=0.05, label="瘦脸强度")
            run_button = gr.Button("开始编辑", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(label="编辑结果", type="numpy")
            status = gr.Textbox(label="状态", interactive=False)

    run_button.click(
        fn=process,
        inputs=[input_image, hair_color, hair_sat, mouth_size, slim_face],
        outputs=[output_image, status],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
