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


def _build_edit_prompt(mouth_scale: float, eye_scale: float, slim_face_strength: float) -> str:
    instructions = ["high-quality portrait", "preserve person identity", "natural skin texture"]
    if mouth_scale > 1.03:
        instructions.append("slightly fuller lips")
    elif mouth_scale < 0.97:
        instructions.append("slightly thinner lips")

    if eye_scale > 1.03:
        instructions.append("larger and brighter eyes")
    elif eye_scale < 0.97:
        instructions.append("slightly smaller eyes")

    if slim_face_strength > 0.05:
        instructions.append("subtly slimmer jawline")

    instructions.append("avoid distortion")
    return ", ".join(instructions)


def _safe_crop_box(landmarks: FaceLandmarks, width: int, height: int) -> tuple[int, int, int, int]:
    oval = landmarks.points[_FACE_OVAL]
    x, y, w, h = cv2.boundingRect(oval.astype(np.int32))
    pad_x = int(w * 0.45)
    pad_top = int(h * 0.65)
    pad_bottom = int(h * 0.45)
    left = max(0, x - pad_x)
    top = max(0, y - pad_top)
    right = min(width, x + w + pad_x)
    bottom = min(height, y + h + pad_bottom)
    return left, top, right, bottom


def _face_mask(image_shape: tuple[int, int, int], landmarks: FaceLandmarks, blur: int = 31) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    oval = landmarks.points[_FACE_OVAL].astype(np.int32)
    cv2.fillConvexPoly(mask, oval, 255)
    if blur % 2 == 0:
        blur += 1
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return (mask.astype(np.float32) / 255.0)[..., None]


def _hair_mask(image_shape: tuple[int, int, int], landmarks: FaceLandmarks) -> np.ndarray:
    h, w = image_shape[:2]
    pts = landmarks.points

    oval = pts[_FACE_OVAL].astype(np.int32)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, oval, 255)

    x, y, fw, fh = cv2.boundingRect(oval)
    hair_region = np.zeros((h, w), dtype=np.uint8)
    top = max(0, y - int(0.70 * fh))
    bottom = y + int(0.45 * fh)
    left = max(0, x - int(0.24 * fw))
    right = min(w, x + fw + int(0.24 * fw))
    hair_region[top:bottom, left:right] = 255

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
    blend = edited.astype(np.float32) * mask + image.astype(np.float32) * (1.0 - mask)
    return np.clip(blend, 0, 255).astype(np.uint8)


def _dense_rbf_warp(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray, face_alpha: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)

    disp = (dst_points - src_points).astype(np.float32)
    movement = np.linalg.norm(disp, axis=1)

    sigma = np.maximum(12.0, movement * 14.0)
    field_x = np.zeros((h, w), dtype=np.float32)
    field_y = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.full((h, w), 1e-6, dtype=np.float32)

    for i in range(src_points.shape[0]):
        px, py = src_points[i]
        dx, dy = disp[i]
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            continue

        d2 = (x - px) ** 2 + (y - py) ** 2
        w_i = np.exp(-d2 / (2.0 * sigma[i] ** 2)).astype(np.float32)

        field_x += w_i * dx
        field_y += w_i * dy
        weight_sum += w_i

    shift_x = (field_x / weight_sum) * face_alpha[..., 0]
    shift_y = (field_y / weight_sum) * face_alpha[..., 0]

    src_x = np.clip(x - shift_x, 0, w - 1).astype(np.float32)
    src_y = np.clip(y - shift_y, 0, h - 1).astype(np.float32)
    return cv2.remap(image, src_x, src_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def _scale_landmark_region(points: np.ndarray, indices: list[int], scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-3:
        return points
    edited = points.copy()
    region = edited[indices]
    center = region.mean(axis=0)
    edited[indices] = center + (region - center) * scale
    return edited


def _slim_landmarks(points: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 1e-4:
        return points

    edited = points.copy()
    nose = edited[_NOSE_TIP]
    for idx in _JAW_LEFT + _JAW_RIGHT:
        edited[idx] = edited[idx] + (nose - edited[idx]) * (0.24 * strength)
    return edited


def _beauty_enhance(image: np.ndarray, landmarks: FaceLandmarks, intensity: float) -> np.ndarray:
    if intensity <= 0:
        return image

    alpha = _face_mask(image.shape, landmarks, blur=35)

    smooth = cv2.bilateralFilter(image, d=0, sigmaColor=35 + int(20 * intensity), sigmaSpace=9 + int(8 * intensity))
    detail = cv2.addWeighted(image, 1.35, smooth, -0.35, 0)
    beauty = cv2.addWeighted(smooth, 0.78 + 0.14 * intensity, detail, 0.22 - 0.10 * intensity, 3)

    lab = cv2.cvtColor(beauty, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] = np.clip(lab[..., 0] * (1.03 + 0.05 * intensity), 0, 255)
    beauty = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    blend = beauty.astype(np.float32) * alpha + image.astype(np.float32) * (1.0 - alpha)
    return np.clip(blend, 0, 255).astype(np.uint8)


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
        self._pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=dtype,
        )
        self._pix2pix = self._pix2pix.to("cuda" if torch.cuda.is_available() else "cpu")
        return True

    def detect(self, bgr_image: np.ndarray) -> Optional[FaceLandmarks]:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        h, w = bgr_image.shape[:2]
        return FaceLandmarks.from_mediapipe(result.multi_face_landmarks[0], w, h)

    def _geometry_edit(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        mouth_scale: float,
        eye_scale: float,
        slim_face_strength: float,
    ) -> np.ndarray:
        src_points = landmarks.points.copy()
        dst_points = src_points.copy()

        dst_points = _scale_landmark_region(dst_points, _MOUTH, mouth_scale)
        dst_points = _scale_landmark_region(dst_points, _LEFT_EYE, eye_scale)
        dst_points = _scale_landmark_region(dst_points, _RIGHT_EYE, eye_scale)
        dst_points = _slim_landmarks(dst_points, slim_face_strength)

        face_alpha = _face_mask(image.shape, landmarks, blur=41)
        warped = _dense_rbf_warp(image, src_points, dst_points, face_alpha)
        return warped

    def _model_edit(
        self,
        image: np.ndarray,
        landmarks: FaceLandmarks,
        mouth_scale: float,
        eye_scale: float,
        slim_face_strength: float,
    ) -> np.ndarray:
        h, w = image.shape[:2]
        l, t, r, b = _safe_crop_box(landmarks, w, h)
        crop = image[t:b, l:r]

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        prompt = _build_edit_prompt(mouth_scale, eye_scale, slim_face_strength)
        result = self._pix2pix(
            prompt=prompt,
            image=Image.fromarray(crop_rgb),
            num_inference_steps=30,
            image_guidance_scale=1.35,
            guidance_scale=7.5,
        ).images[0]

        edited_crop = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        if edited_crop.shape[:2] != crop.shape[:2]:
            edited_crop = cv2.resize(edited_crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)

        output = image.copy()
        center = ((l + r) // 2, (t + b) // 2)
        mask = np.full(crop.shape[:2], 255, dtype=np.uint8)
        return cv2.seamlessClone(edited_crop, output, mask, center, cv2.NORMAL_CLONE)

    def edit(
        self,
        bgr_image: np.ndarray,
        hair_hue_shift: float = 0.0,
        hair_saturation_boost: float = 0.0,
        mouth_scale: float = 1.0,
        eye_scale: float = 1.0,
        slim_face_strength: float = 0.0,
    ) -> tuple[np.ndarray, str]:
        landmarks = self.detect(bgr_image)
        if landmarks is None:
            return bgr_image, "未检测到人脸，请尝试更清晰的正脸图片。"

        hair_saturation_boost = _clamp(hair_saturation_boost, 0.0, 1.0)
        mouth_scale = _clamp(mouth_scale, 0.70, 1.55)
        eye_scale = _clamp(eye_scale, 0.80, 1.45)
        slim_face_strength = _clamp(slim_face_strength, 0.0, 1.0)

        model_used = False
        try:
            if self._ensure_model():
                output = self._model_edit(bgr_image, landmarks, mouth_scale, eye_scale, slim_face_strength)
                model_used = True
            else:
                output = self._geometry_edit(bgr_image, landmarks, mouth_scale, eye_scale, slim_face_strength)
        except Exception:
            output = self._geometry_edit(bgr_image, landmarks, mouth_scale, eye_scale, slim_face_strength)

        output = _beauty_enhance(output, landmarks, intensity=0.72)
        mask = _hair_mask(output.shape, landmarks)
        output = _shift_hair_color(output, mask, hair_hue_shift, hair_saturation_boost)

        applied = (
            f"发色偏移={hair_hue_shift:.0f}, 发色饱和度={hair_saturation_boost:.2f}, "
            f"嘴巴={mouth_scale:.2f}, 眼睛={eye_scale:.2f}, 瘦脸={slim_face_strength:.2f}"
        )
        if model_used:
            return output, f"编辑完成（大模型+高精度位移场）。{applied}"
        return output, f"编辑完成（高精度位移场美颜；安装 diffusers+torch 可启用大模型）。{applied}"
