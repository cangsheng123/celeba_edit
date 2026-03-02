# CelebA 人脸编辑系统（桌面 App）

这是一个**非浏览器**的人脸编辑系统，直接运行桌面应用即可使用，支持：

- 头发颜色调整
- 嘴巴大小调整
- 眼睛大小调整
- 瘦脸强度调整
- 编辑结果保存

## 快速启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

运行后会打开桌面窗口（Tkinter），无需浏览器。

## 系统组成

- `app.py`：桌面 App 界面（上传、参数调节、应用编辑、保存结果）
- `face_editor.py`：人脸关键点检测与图像编辑算法
- `requirements.txt`：依赖列表

## 功能流程

1. 上传人脸图像。
2. 调整发色、发色饱和度、嘴巴大小、眼睛大小、瘦脸强度。
3. 点击“应用编辑”实时生成结果。
4. 点击“保存结果”导出图片。

## 技术方案

- 感知层：MediaPipe Face Mesh 提取人脸关键点。
- 编辑层：OpenCV 局部几何形变 + HSV 发色映射。
- 交互层：Tkinter + Pillow 桌面可视化界面。
