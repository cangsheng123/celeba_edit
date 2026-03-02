from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

try:
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline
except Exception:  # optional runtime dependency
    torch = None
    StableDiffusionInstructPix2PixPipeline = None


_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
_MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
_LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145]
_RIGHT_EYE = [263, 362, 385, 386, 387, 388, 466, 373, 374, 380]
_JAW_LEFT = [234, 93, 132, 58, 172, 136, 149, 150]
_JAW_RIGHT = [454, 323, 361, 288, 397, 365, 379, 378]
_NOSE_WING_LEFT = [114, 188]
_NOSE_WING_RIGHT = [343, 412]
_CHIN = [152]
_NOSE_TIP = 1


@dataclass
class FaceLandmarks:
    points: np.ndarray

    @classmethod
    def from_mediapipe(cls, face_landmarks, width: int, height: int) -> "FaceLandmarks":
        pts = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks.landmark], dtype=np.float32)
        return cls(points=pts)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _face_alpha(image_shape: tuple[int, int, int], landmarks: FaceLandmarks, blur: int = 41) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, landmarks.points[_FACE_OVAL].astype(np.int32), 255)
    if blur % 2 == 0:
        blur += 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return (mask.astype(np.float32) / 255.0)[..., None]


def _hair_mask(image_shape: tuple[int, int, int], landmarks: FaceLandmarks) -> np.ndarray:
    h, w = image_shape[:2]
    oval = landmarks.points[_FACE_OVAL].astype(np.int32)

    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, oval, 255)

    x, y, fw, fh = cv2.boundingRect(oval)
    hair_region = np.zeros((h, w), dtype=np.uint8)
    hair_region[max(0, y - int(0.72 * fh)): y + int(0.45 * fh), max(0, x - int(0.24 * fw)): min(w, x + fw + int(0.24 * fw))] = 255

    hair_mask = cv2.bitwise_and(hair_region, cv2.bitwise_not(face_mask))
    return cv2.GaussianBlur(hair_mask, (27, 27), 0)


def _shift_hair_color(image: np.ndarray, hair_mask: np.ndarray, hue_shift: float, sat_boost: float) -> np.ndarray:
    if abs(hue_shift) < 1e-3 and abs(sat_boost) < 1e-3:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask = (hair_mask.astype(np.float32) / 255.0)[..., None]

    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat_boost), 0, 255)

    edited = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.clip(edited.astype(np.float32) * mask + image.astype(np.float32) * (1.0 - mask), 0, 255).astype(np.uint8)


def _build_prompt(mouth_scale: float, eye_scale: float, slim: float, nose_scale: float, chin_scale: float) -> str:
    prompt = ["high quality realistic portrait", "preserve identity", "avoid artifacts"]
    if eye_scale > 1.02:
        prompt.append("slightly bigger eyes")
    elif eye_scale < 0.98:
        prompt.append("slightly smaller eyes")
    if mouth_scale > 1.02:
        prompt.append("slightly fuller lips")
    elif mouth_scale < 0.98:
        prompt.append("slightly thinner lips")
    if slim > 0.05:
        prompt.append("slimmer jawline")
    if nose_scale < 0.99:
        prompt.append("slightly narrower nose")
    if chin_scale > 1.01:
        prompt.append("slightly longer chin")
    elif chin_scale < 0.99:
        prompt.append("slightly shorter chin")
    return ", ".join(prompt)


def _safe_crop_box(landmarks: FaceLandmarks, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(landmarks.points[_FACE_OVAL].astype(np.int32))
    px, ptop, pbot = int(w * 0.45), int(h * 0.65), int(h * 0.45)
    return max(0, x - px), max(0, y - ptop), min(width, x + w + px), min(height, y + h + pbot)


def _scale_region(points: np.ndarray, indices: list[int], scale: float) -> None:
    if abs(scale - 1.0) < 1e-4:
        return
    center = points[indices].mean(axis=0)
    points[indices] = center + (points[indices] - center) * scale


def _slim_points(points: np.ndarray, strength: float) -> None:
    if strength <= 1e-4:
        return
    nose = points[_NOSE_TIP]
    for idx in _JAW_LEFT + _JAW_RIGHT:
        points[idx] = points[idx] + (nose - points[idx]) * (0.22 * strength)


def _reshape_points(src_points: np.ndarray, mouth_scale: float, eye_scale: float, slim: float, nose_scale: float, chin_scale: float) -> np.ndarray:
    dst = src_points.copy()
    _scale_region(dst, _MOUTH, mouth_scale)
    _scale_region(dst, _LEFT_EYE, eye_scale)
    _scale_region(dst, _RIGHT_EYE, eye_scale)
    _scale_region(dst, _NOSE_WING_LEFT + _NOSE_WING_RIGHT, nose_scale)
    _slim_points(dst, slim)

    if abs(chin_scale - 1.0) > 1e-4:
        nose = dst[_NOSE_TIP]
        chin = dst[_CHIN[0]]
        dst[_CHIN[0]] = nose + (chin - nose) * chin_scale

    return dst


def _dense_warp(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)

    disp = dst_points - src_points
    movement = np.linalg.norm(disp, axis=1)
    sigma = np.maximum(14.0, movement * 16.0)

    sum_dx = np.zeros((h, w), dtype=np.float32)
    sum_dy = np.zeros((h, w), dtype=np.float32)
    sum_w = np.full((h, w), 1e-6, dtype=np.float32)

    for i, (px, py) in enumerate(src_points):
        dx, dy = disp[i]
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            continue
        d2 = (x - px) ** 2 + (y - py) ** 2
        wi = np.exp(-d2 / (2.0 * sigma[i] * sigma[i])).astype(np.float32)
        sum_dx += wi * dx
        sum_dy += wi * dy
        sum_w += wi

    shift_x = (sum_dx / sum_w) * alpha[..., 0]
    shift_y = (sum_dy / sum_w) * alpha[..., 0]

    src_x = np.clip(x - shift_x, 0, w - 1).astype(np.float32)
    src_y = np.clip(y - shift_y, 0, h - 1).astype(np.float32)
    return cv2.remap(image, src_x, src_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def _beautify(image: np.ndarray, landmarks: FaceLandmarks, intensity: float = 0.7) -> np.ndarray:
    alpha = _face_alpha(image.shape, landmarks, blur=35)
    smooth = cv2.bilateralFilter(image, d=0, sigmaColor=36 + int(16 * intensity), sigmaSpace=10 + int(6 * intensity))
    detail = cv2.addWeighted(image, 1.3, smooth, -0.3, 0)
    beauty = cv2.addWeighted(smooth, 0.76 + 0.12 * intensity, detail, 0.24 - 0.08 * intensity, 2)

    lab = cv2.cvtColor(beauty, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] = np.clip(lab[..., 0] * (1.03 + 0.04 * intensity), 0, 255)
    beauty = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return np.clip(beauty.astype(np.float32) * alpha + image.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def _hd_upscale(image: np.ndarray, upscale: float) -> np.ndarray:
    if upscale <= 1.0:
        return image
    h, w = image.shape[:2]
    nh, nw = int(h * upscale), int(w * upscale)
    up = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), 1.0)
    sharp = cv2.addWeighted(up, 1.35, blur, -0.35, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


class FaceEditor:
    def __init__(self) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self._pix2pix = None
        self._pix2pix_available = StableDiffusionInstructPix2PixPipeline is not None and torch is not None

    def _ensure_model(self) -> bool:
        if not self._pix2pix_available:
            return False
        if self._pix2pix is not None:
            return True
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=dtype)
        self._pix2pix = self._pix2pix.to("cuda" if torch.cuda.is_available() else "cpu")
        return True

    def detect(self, bgr_image: np.ndarray) -> Optional[FaceLandmarks]:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        h, w = bgr_image.shape[:2]
        return FaceLandmarks.from_mediapipe(result.multi_face_landmarks[0], w, h)

    def _model_edit(self, image: np.ndarray, landmarks: FaceLandmarks, mouth_scale: float, eye_scale: float, slim: float, nose_scale: float, chin_scale: float) -> np.ndarray:
        h, w = image.shape[:2]
        l, t, r, b = _safe_crop_box(landmarks, w, h)
        crop = image[t:b, l:r]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        prompt = _build_prompt(mouth_scale, eye_scale, slim, nose_scale, chin_scale)

        result = self._pix2pix(
            prompt=prompt,
            image=Image.fromarray(crop_rgb),
            num_inference_steps=32,
            image_guidance_scale=1.4,
            guidance_scale=7.5,
        ).images[0]
        edited = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        if edited.shape[:2] != crop.shape[:2]:
            edited = cv2.resize(edited, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)

        out = image.copy()
        mask = np.full(crop.shape[:2], 255, dtype=np.uint8)
        return cv2.seamlessClone(edited, out, mask, ((l + r) // 2, (t + b) // 2), cv2.NORMAL_CLONE)

    def _precision_geometry_edit(self, image: np.ndarray, landmarks: FaceLandmarks, mouth_scale: float, eye_scale: float, slim: float, nose_scale: float, chin_scale: float) -> np.ndarray:
        src = landmarks.points.copy()
        dst = _reshape_points(src, mouth_scale, eye_scale, slim, nose_scale, chin_scale)
        alpha = _face_alpha(image.shape, landmarks, blur=43)
        return _dense_warp(image, src, dst, alpha)

    def edit(
        self,
        bgr_image: np.ndarray,
        hair_hue_shift: float = 0.0,
        hair_saturation_boost: float = 0.0,
        mouth_scale: float = 1.0,
        eye_scale: float = 1.0,
        slim_face_strength: float = 0.0,
        nose_scale: float = 1.0,
        chin_scale: float = 1.0,
        hd_upscale: float = 1.0,
    ) -> tuple[np.ndarray, str]:
        landmarks = self.detect(bgr_image)
        if landmarks is None:
            return bgr_image, "未检测到人脸，请尝试更清晰的正脸图片。"

        hair_saturation_boost = _clamp(hair_saturation_boost, 0.0, 1.0)
        mouth_scale = _clamp(mouth_scale, 0.75, 1.35)
        eye_scale = _clamp(eye_scale, 0.85, 1.30)
        slim_face_strength = _clamp(slim_face_strength, 0.0, 1.0)
        nose_scale = _clamp(nose_scale, 0.85, 1.15)
        chin_scale = _clamp(chin_scale, 0.90, 1.20)
        hd_upscale = _clamp(hd_upscale, 1.0, 2.0)

        model_used = False
        try:
            if self._ensure_model():
                output = self._model_edit(bgr_image, landmarks, mouth_scale, eye_scale, slim_face_strength, nose_scale, chin_scale)
                model_used = True
            else:
                output = self._precision_geometry_edit(bgr_image, landmarks, mouth_scale, eye_scale, slim_face_strength, nose_scale, chin_scale)
        except Exception:
            output = self._precision_geometry_edit(bgr_image, landmarks, mouth_scale, eye_scale, slim_face_strength, nose_scale, chin_scale)

        output = _beautify(output, landmarks, intensity=0.72)
        output = _shift_hair_color(output, _hair_mask(output.shape, landmarks), hair_hue_shift, hair_saturation_boost)
        output = _hd_upscale(output, hd_upscale)

        mode = "大模型+精控位移场" if model_used else "精控位移场（可安装diffusers+torch启用大模型）"
        applied = (
            f"发色={hair_hue_shift:.0f}, 饱和度={hair_saturation_boost:.2f}, 嘴巴={mouth_scale:.2f}, "
            f"眼睛={eye_scale:.2f}, 瘦脸={slim_face_strength:.2f}, 鼻翼={nose_scale:.2f}, 下巴={chin_scale:.2f}, 高清={hd_upscale:.2f}x"
        )
        return output, f"编辑完成（{mode}）。{applied}"
