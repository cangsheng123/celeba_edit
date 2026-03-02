from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import cv2
import mediapipe as mp
import numpy as np


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


@dataclass
class EditOptions:
    hair_hue_shift: float
    hair_saturation_boost: float
    mouth_scale: float
    eye_scale: float
    slim_face_strength: float
    nose_scale: float
    chin_scale: float
    hd_upscale: float


class Processor(Protocol):
    def process(self, frame: np.ndarray, landmarks: FaceLandmarks, options: EditOptions) -> np.ndarray:
        ...


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


def _scale_region(points: np.ndarray, indices: list[int], scale: float) -> None:
    if abs(scale - 1.0) < 1e-4:
        return
    region = points[indices]
    center = region.mean(axis=0)
    points[indices] = center + (region - center) * scale


def _dense_warp(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)

    displacement = dst_points - src_points
    sigma = np.maximum(14.0, np.linalg.norm(displacement, axis=1) * 16.0)

    sum_dx = np.zeros((h, w), dtype=np.float32)
    sum_dy = np.zeros((h, w), dtype=np.float32)
    sum_w = np.full((h, w), 1e-6, dtype=np.float32)

    for i, (px, py) in enumerate(src_points):
        dx, dy = displacement[i]
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


class GeometryProcessor:
    """FaceFusion-style step processor: applies landmark-driven shape edits."""

    def process(self, frame: np.ndarray, landmarks: FaceLandmarks, options: EditOptions) -> np.ndarray:
        src = landmarks.points.copy()
        dst = src.copy()

        _scale_region(dst, _MOUTH, options.mouth_scale)
        _scale_region(dst, _LEFT_EYE, options.eye_scale)
        _scale_region(dst, _RIGHT_EYE, options.eye_scale)
        _scale_region(dst, _NOSE_WING_LEFT + _NOSE_WING_RIGHT, options.nose_scale)

        nose = dst[_NOSE_TIP]
        for idx in _JAW_LEFT + _JAW_RIGHT:
            dst[idx] = dst[idx] + (nose - dst[idx]) * (0.22 * options.slim_face_strength)

        chin = dst[_CHIN[0]]
        dst[_CHIN[0]] = nose + (chin - nose) * options.chin_scale

        alpha = _face_alpha(frame.shape, landmarks, blur=43)
        return _dense_warp(frame, src, dst, alpha)


class BeautyProcessor:
    def process(self, frame: np.ndarray, landmarks: FaceLandmarks, options: EditOptions) -> np.ndarray:
        alpha = _face_alpha(frame.shape, landmarks, blur=35)
        smooth = cv2.bilateralFilter(frame, d=0, sigmaColor=48, sigmaSpace=12)
        detail = cv2.addWeighted(frame, 1.30, smooth, -0.30, 0)
        beauty = cv2.addWeighted(smooth, 0.82, detail, 0.18, 2)

        lab = cv2.cvtColor(beauty, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] = np.clip(lab[..., 0] * 1.06, 0, 255)
        beauty = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return np.clip(beauty.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


class HairColorProcessor:
    def _hair_mask(self, image_shape: tuple[int, int, int], landmarks: FaceLandmarks) -> np.ndarray:
        h, w = image_shape[:2]
        oval = landmarks.points[_FACE_OVAL].astype(np.int32)

        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, oval, 255)

        x, y, fw, fh = cv2.boundingRect(oval)
        hair_region = np.zeros((h, w), dtype=np.uint8)
        hair_region[max(0, y - int(0.72 * fh)): y + int(0.45 * fh), max(0, x - int(0.24 * fw)): min(w, x + fw + int(0.24 * fw))] = 255

        return cv2.GaussianBlur(cv2.bitwise_and(hair_region, cv2.bitwise_not(face_mask)), (27, 27), 0)

    def process(self, frame: np.ndarray, landmarks: FaceLandmarks, options: EditOptions) -> np.ndarray:
        if abs(options.hair_hue_shift) < 1e-3 and abs(options.hair_saturation_boost) < 1e-3:
            return frame

        hair_mask = self._hair_mask(frame.shape, landmarks)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        mask = (hair_mask.astype(np.float32) / 255.0)[..., None]

        hsv[..., 0] = (hsv[..., 0] + options.hair_hue_shift) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + options.hair_saturation_boost), 0, 255)

        edited = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return np.clip(edited.astype(np.float32) * mask + frame.astype(np.float32) * (1.0 - mask), 0, 255).astype(np.uint8)


class UpscaleProcessor:
    def process(self, frame: np.ndarray, landmarks: FaceLandmarks, options: EditOptions) -> np.ndarray:
        if options.hd_upscale <= 1.0:
            return frame
        h, w = frame.shape[:2]
        nh, nw = int(h * options.hd_upscale), int(w * options.hd_upscale)
        up = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(up, (0, 0), 1.0)
        return np.clip(cv2.addWeighted(up, 1.35, blur, -0.35, 0), 0, 255).astype(np.uint8)


class FaceEditor:
    """FaceFusion-like pipeline orchestration with modular frame processors."""

    def __init__(self) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self._processors: list[Processor] = [
            GeometryProcessor(),
            BeautyProcessor(),
            HairColorProcessor(),
            UpscaleProcessor(),
        ]

    def detect(self, bgr_image: np.ndarray) -> Optional[FaceLandmarks]:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        h, w = bgr_image.shape[:2]
        return FaceLandmarks.from_mediapipe(result.multi_face_landmarks[0], w, h)

    def _build_options(
        self,
        hair_hue_shift: float,
        hair_saturation_boost: float,
        mouth_scale: float,
        eye_scale: float,
        slim_face_strength: float,
        nose_scale: float,
        chin_scale: float,
        hd_upscale: float,
    ) -> EditOptions:
        return EditOptions(
            hair_hue_shift=hair_hue_shift,
            hair_saturation_boost=_clamp(hair_saturation_boost, 0.0, 1.0),
            mouth_scale=_clamp(mouth_scale, 0.75, 1.35),
            eye_scale=_clamp(eye_scale, 0.85, 1.30),
            slim_face_strength=_clamp(slim_face_strength, 0.0, 1.0),
            nose_scale=_clamp(nose_scale, 0.85, 1.15),
            chin_scale=_clamp(chin_scale, 0.90, 1.20),
            hd_upscale=_clamp(hd_upscale, 1.0, 2.0),
        )

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

        options = self._build_options(
            hair_hue_shift,
            hair_saturation_boost,
            mouth_scale,
            eye_scale,
            slim_face_strength,
            nose_scale,
            chin_scale,
            hd_upscale,
        )

        output = bgr_image.copy()
        for processor in self._processors:
            output = processor.process(output, landmarks, options)

        applied = (
            f"发色={options.hair_hue_shift:.0f}, 饱和度={options.hair_saturation_boost:.2f}, 嘴巴={options.mouth_scale:.2f}, "
            f"眼睛={options.eye_scale:.2f}, 瘦脸={options.slim_face_strength:.2f}, 鼻翼={options.nose_scale:.2f}, "
            f"下巴={options.chin_scale:.2f}, 高清={options.hd_upscale:.2f}x"
        )
        return output, f"编辑完成（FaceFusion 风格处理管线）。{applied}"
