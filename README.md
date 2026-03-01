# CelebA 人脸编辑系统（原型）

这是一个面向 **CelebA 人脸图像** 的轻量编辑原型，提供以下常见“批图软件”能力：

- 头发颜色调整
- 嘴巴大小调整
- 瘦脸强度调整

> 当前方案采用传统视觉（MediaPipe 人脸关键点 + OpenCV 局部形变/颜色映射）实现，适合快速搭建可交互演示界面。

## 快速启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

浏览器访问：`http://localhost:7860`

## 项目结构

- `app.py`: Gradio 前端界面与交互逻辑
- `face_editor.py`: 人脸检测、关键点定位、发色/嘴巴/瘦脸编辑算法
- `requirements.txt`: 依赖

## 后续升级建议（真正“CelebA 级”可控编辑）

1. 使用 CelebA / CelebA-HQ 训练 StyleGAN 或扩散模型。
2. 基于属性方向（hair color/smile/age）做 latent editing。
3. 用分割模型（例如 BiSeNet）替代启发式头发区域，提升发色编辑稳定性。
4. 增加多属性联动约束，减少属性冲突。

