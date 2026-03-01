from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
              379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
              234, 127, 162, 21, 54, 103, 67, 109]
_MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
          324, 318, 402, 317, 14, 87, 178, 88, 95]
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454


@dataclass
class FaceLandmarks:
    points: np.ndarray

    @classmethod
    def from_mediapipe(cls, face_landmarks, width: int, height: int) -> "FaceLandmarks":
        pts = np.array(
            [[lm.x * width, lm.y * height] for lm in face_landmarks.landmark],
            dtype=np.float32,
        )
        return cls(points=pts)


def _local_scale_warp(image: np.ndarray, center: tuple[float, float], radius: float, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    cx, cy = center
    dx = x - cx
    dy = y - cy
    dist = np.sqrt(dx * dx + dy * dy)

    mask = dist < radius
    safe_dist = np.maximum(dist, 1e-6)

    factor = np.ones_like(dist, dtype=np.float32)
    factor[mask] = 1 - (1 - 1 / scale) * (1 - dist[mask] / radius) ** 2

    src_x = cx + dx * factor
    src_y = cy + dy * factor

    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)
    return cv2.remap(image, src_x.astype(np.float32), src_y.astype(np.float32), cv2.INTER_LINEAR)


def _slim_face_warp(image: np.ndarray, landmarks: FaceLandmarks, strength: float) -> np.ndarray:
    if strength <= 0:
        return image

    h, w = image.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    pts = landmarks.points

    left = pts[_LEFT_CHEEK]
    right = pts[_RIGHT_CHEEK]
    face_center_x = (left[0] + right[0]) / 2
    radius = np.linalg.norm(right - left) * 0.35

    sigma2 = (radius ** 2) + 1e-6
    pull = strength * radius * 0.35

    def gaussian_pull(cx: float, cy: float, direction: float):
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        return direction * pull * np.exp(-d2 / (2 * sigma2))

    shift_x = gaussian_pull(left[0], left[1], +1.0) + gaussian_pull(right[0], right[1], -1.0)

    src_x = np.clip(x + shift_x, 0, w - 1)
    src_y = np.clip(y, 0, h - 1)
    return cv2.remap(image, src_x.astype(np.float32), src_y.astype(np.float32), cv2.INTER_LINEAR)


def _hair_mask(image_shape: tuple[int, int, int], landmarks: FaceLandmarks) -> np.ndarray:
    h, w = image_shape[:2]
    pts = landmarks.points

    oval = pts[_FACE_OVAL].astype(np.int32)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, oval, 255)

    x, y, fw, fh = cv2.boundingRect(oval)
    hair_region = np.zeros((h, w), dtype=np.uint8)
    top = max(0, y - int(0.55 * fh))
    bottom = y + int(0.35 * fh)
    left = max(0, x - int(0.18 * fw))
    right = min(w, x + fw + int(0.18 * fw))
    hair_region[top:bottom, left:right] = 255

    hair_mask = cv2.bitwise_and(hair_region, cv2.bitwise_not(face_mask))
    hair_mask = cv2.GaussianBlur(hair_mask, (21, 21), 0)
    return hair_mask


def _shift_hair_color(image: np.ndarray, hair_mask: np.ndarray, hue_shift: float, sat_boost: float) -> np.ndarray:
    if abs(hue_shift) < 1e-3 and abs(sat_boost) < 1e-3:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask = (hair_mask.astype(np.float32) / 255.0)[..., None]

    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat_boost), 0, 255)

    edited = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    blend = edited.astype(np.float32) * mask + image.astype(np.float32) * (1 - mask)
    return np.clip(blend, 0, 255).astype(np.uint8)


class FaceEditor:
    def __init__(self) -> None:
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

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
        slim_face_strength: float = 0.0,
    ) -> tuple[np.ndarray, str]:
        landmarks = self.detect(bgr_image)
        if landmarks is None:
            return bgr_image, "未检测到人脸，请尝试更清晰的正脸图片。"

        output = bgr_image.copy()

        # 瘦脸
        output = _slim_face_warp(output, landmarks, slim_face_strength)

        # 嘴巴大小
        mouth_pts = landmarks.points[_MOUTH]
        mouth_center = mouth_pts.mean(axis=0)
        mouth_radius = max(np.linalg.norm(mouth_pts.max(axis=0) - mouth_pts.min(axis=0)) * 0.9, 1.0)
        output = _local_scale_warp(output, (mouth_center[0], mouth_center[1]), mouth_radius, mouth_scale)

        # 头发颜色
        mask = _hair_mask(output.shape, landmarks)
        output = _shift_hair_color(output, mask, hair_hue_shift, hair_saturation_boost)

        return output, "编辑完成（基于人脸关键点与局部形变）。"
