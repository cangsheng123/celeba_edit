from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from face_editor import FaceEditor

editor = FaceEditor()


HAIR_COLOR_PRESETS = {
    "保持不变": 0,
    "金色": 18,
    "红棕": 8,
    "蓝黑": 105,
    "紫色": 145,
}


def process(
    image: np.ndarray,
    hair_color: str,
    hair_sat_boost: float,
    mouth_size: float,
    eye_size: float,
    slim_face: float,
):
    if image is None:
        return None, "请先上传图片。"

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    out, msg = editor.edit(
        bgr,
        hair_hue_shift=HAIR_COLOR_PRESETS.get(hair_color, 0),
        hair_saturation_boost=hair_sat_boost,
        mouth_scale=mouth_size,
        eye_scale=eye_size,
        slim_face_strength=slim_face,
    )
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return rgb, msg


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="rose", neutral_hue="slate"),
    title="CelebA 人脸编辑系统",
    css="""
    .container {max-width: 1180px; margin: 0 auto;}
    .hero-card {
        border-radius: 16px;
        padding: 18px 22px;
        background: linear-gradient(120deg, #fff1f2 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        margin-bottom: 8px;
    }
    .hero-card h1 {font-size: 2.0rem; margin: 0 0 8px 0;}
    .hero-card p {margin: 0; opacity: 0.9;}
    """,
) as demo:
    gr.Markdown(
        """
<div class="hero-card">
  <h1>✨ CelebA 人脸编辑系统</h1>
  <p>上传一张人脸图片，即可进行发色、嘴巴、眼睛、瘦脸等 P 图编辑。适合做课程演示和快速原型。</p>
</div>
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传人脸", type="numpy")
            hair_color = gr.Dropdown(
                label="头发颜色", choices=list(HAIR_COLOR_PRESETS.keys()), value="保持不变"
            )
            hair_sat = gr.Slider(0, 1.0, value=0.35, step=0.05, label="头发饱和度增强")
            mouth_size = gr.Slider(0.75, 1.35, value=1.0, step=0.01, label="嘴巴大小")
            eye_size = gr.Slider(0.8, 1.4, value=1.0, step=0.01, label="眼睛大小")
            slim_face = gr.Slider(0, 1.0, value=0.35, step=0.05, label="瘦脸强度")
            with gr.Row():
                run_button = gr.Button("开始编辑", variant="primary")
                clear_button = gr.Button("清空")
            status = gr.Textbox(label="状态", interactive=False)

        with gr.Column(scale=1):
            output_image = gr.Image(label="编辑结果", type="numpy")
            with gr.Accordion("系统说明", open=False):
                gr.Markdown(
                    """
- 感知模块：MediaPipe Face Mesh 定位关键点。
- 编辑模块：OpenCV 局部形变 + HSV 发色映射。
- 支持参数：发色、发色饱和度、嘴巴大小、眼睛大小、瘦脸强度。
"""
                )

    run_button.click(
        fn=process,
        inputs=[input_image, hair_color, hair_sat, mouth_size, eye_size, slim_face],
        outputs=[output_image, status],
    )

    clear_button.click(
        fn=lambda: (None, None, "已清空"),
        inputs=[],
        outputs=[input_image, output_image, status],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
