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
- 编辑层：FaceFusion 风格处理管线（几何处理器 + 美颜处理器 + 发色处理器 + 高清处理器）。
- 交互层：Tkinter + Pillow 桌面可视化界面。


## Stable Diffusion WebUI 精准人脸版（推荐）

如果你希望达到更接近专业 P 图软件的精修效果，请使用新增的 `ai_face_app.py`（通过 WebUI API + ControlNet Depth）。

### 1) 启动 WebUI API
在你的 Stable Diffusion WebUI 启动参数中添加：

```bash
--api --listen
```

### 2) 安装 ControlNet
- 扩展安装：`sd-webui-controlnet`
- 模型放置：`extensions/sd-webui-controlnet/models`
- 推荐模型：`control_v11f1p_sd15_depth`

### 3) 运行专属界面
```bash
python ai_face_app.py
```

界面内包含：
- 一键上传
- 魔法滑杆（眼睛、嘴唇、年轻化、皮肤、鼻翼）
- ControlNet 锁脸（尽量保持脸型与关键布局）
- 高清参数（steps/cfg/denoise/control weight）
