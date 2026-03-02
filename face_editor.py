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
_JAW_LEFT = [234, 93, 132, 58, 172, 136]
_JAW_RIGHT = [454, 323, 361, 288, 397, 365]
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
    instructions = ["high-quality portrait photo", "keep identity", "keep skin texture natural"]

    if mouth_scale > 1.03:
        instructions.append("slightly larger lips")
    elif mouth_scale < 0.97:
        instructions.append("slightly smaller lips")

    if eye_scale > 1.03:
        instructions.append("larger eyes")
    elif eye_scale < 0.97:
        instructions.append("smaller eyes")

    if slim_face_strength > 0.05:
        instructions.append("slimmer face shape")

    instructions.append("avoid cartoon, avoid deformation")
    return ", ".join(instructions)


def _safe_crop_box(landmarks: FaceLandmarks, width: int, height: int) -> tuple[int, int, int, int]:
    oval = landmarks.points[_FACE_OVAL]
    x, y, w, h = cv2.boundingRect(oval.astype(np.int32))

    pad_x = int(w * 0.45)
    pad_top = int(h * 0.65)
    pad_bottom = int(h * 0.40)

    left = max(0, x - pad_x)
    top = max(0, y - pad_top)
    right = min(width, x + w + pad_x)
    bottom = min(height, y + h + pad_bottom)
    return left, top, right, bottom


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


def _geometric_fallback(image: np.ndarray, landmarks: FaceLandmarks, mouth_scale: float, eye_scale: float, slim: float) -> np.ndarray:
    points = landmarks.points.copy()
    dst = points.copy()

    def scale_region(indices: list[int], s: float) -> None:
        if abs(s - 1.0) < 1e-3:
            return
        region = dst[indices]
        center = region.mean(axis=0)
        dst[indices] = center + (region - center) * s

    scale_region(_MOUTH, mouth_scale)
    scale_region(_LEFT_EYE, eye_scale)
    scale_region(_RIGHT_EYE, eye_scale)

    nose = dst[_NOSE_TIP]
    for idx in _JAW_LEFT + _JAW_RIGHT:
        dst[idx] = dst[idx] + (nose - dst[idx]) * (0.22 * slim)

    rect = (0, 0, image.shape[1], image.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    border = np.array([[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]], dtype=np.float32)
    src_all = np.vstack([points, border])
    dst_all = np.vstack([dst, border])
    for p in src_all:
        subdiv.insert((float(p[0]), float(p[1])))

    out = image.copy()
    for t in subdiv.getTriangleList():
        tri = np.array([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]], dtype=np.float32)
        if np.any(tri[:, 0] < 0) or np.any(tri[:, 0] >= image.shape[1]) or np.any(tri[:, 1] < 0) or np.any(tri[:, 1] >= image.shape[0]):
            continue
        ids = [int(np.argmin(np.linalg.norm(src_all - p, axis=1))) for p in tri]
        if len(set(ids)) < 3:
            continue
        src_tri = src_all[ids].astype(np.float32)
        dst_tri = dst_all[ids].astype(np.float32)

        r1 = cv2.boundingRect(src_tri)
        r2 = cv2.boundingRect(dst_tri)
        src_rect = np.array([[src_tri[i][0] - r1[0], src_tri[i][1] - r1[1]] for i in range(3)], dtype=np.float32)
        dst_rect = np.array([[dst_tri[i][0] - r2[0], dst_tri[i][1] - r2[1]] for i in range(3)], dtype=np.float32)
        patch = image[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        if patch.size == 0 or r2[2] <= 0 or r2[3] <= 0:
            continue
        mat = cv2.getAffineTransform(src_rect, dst_rect)
        warped = cv2.warpAffine(patch, mat, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(dst_rect), (1.0, 1.0, 1.0), lineType=cv2.LINE_AA)
        target = out[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]].astype(np.float32)
        out[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = np.clip(target * (1 - mask) + warped.astype(np.float32) * mask, 0, 255).astype(np.uint8)

    return out


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
        if torch.cuda.is_available():
            self._pix2pix = self._pix2pix.to("cuda")
        else:
            self._pix2pix = self._pix2pix.to("cpu")
        return True

    def detect(self, bgr_image: np.ndarray) -> Optional[FaceLandmarks]:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        h, w = bgr_image.shape[:2]
        return FaceLandmarks.from_mediapipe(result.multi_face_landmarks[0], w, h)

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
        mouth_scale = _clamp(mouth_scale, 0.60, 1.90)
        eye_scale = _clamp(eye_scale, 0.70, 1.90)
        slim_face_strength = _clamp(slim_face_strength, 0.0, 1.0)

        output: np.ndarray
        model_used = False

        try:
            if self._ensure_model():
                h, w = bgr_image.shape[:2]
                l, t, r, b = _safe_crop_box(landmarks, w, h)
                crop = bgr_image[t:b, l:r]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                prompt = _build_edit_prompt(mouth_scale, eye_scale, slim_face_strength)

                image_pil = Image.fromarray(crop_rgb)
                result = self._pix2pix(
                    prompt=prompt,
                    image=image_pil,
                    num_inference_steps=30,
                    image_guidance_scale=1.3,
                    guidance_scale=7.5,
                ).images[0]

                edited_crop = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                if edited_crop.shape[:2] != crop.shape[:2]:
                    edited_crop = cv2.resize(edited_crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)

                output = bgr_image.copy()
                center = ((l + r) // 2, (t + b) // 2)
                mask = np.full(crop.shape[:2], 255, dtype=np.uint8)
                output = cv2.seamlessClone(edited_crop, output, mask, center, cv2.NORMAL_CLONE)
                model_used = True
            else:
                output = _geometric_fallback(bgr_image, landmarks, mouth_scale, eye_scale, slim_face_strength)
        except Exception:
            output = _geometric_fallback(bgr_image, landmarks, mouth_scale, eye_scale, slim_face_strength)

        mask = _hair_mask(output.shape, landmarks)
        output = _shift_hair_color(output, mask, hair_hue_shift, hair_saturation_boost)

        applied = (
            f"发色偏移={hair_hue_shift:.0f}, 发色饱和度={hair_saturation_boost:.2f}, "
            f"嘴巴={mouth_scale:.2f}, 眼睛={eye_scale:.2f}, 瘦脸={slim_face_strength:.2f}"
        )
        if model_used:
            return output, f"编辑完成（大模型 InstructPix2Pix）。{applied}"
        return output, f"编辑完成（几何回退；安装 diffusers+torch 可启用大模型）。{applied}"
