from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


def _add_border_points(width: int, height: int) -> np.ndarray:
    return np.array(
        [
            [0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1],
            [width // 2, 0], [width // 2, height - 1], [0, height // 2], [width - 1, height // 2],
        ],
        dtype=np.float32,
    )


def _delaunay_triangles(points: np.ndarray, width: int, height: int) -> list[tuple[int, int, int]]:
    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))

    triangle_list = subdiv.getTriangleList()
    triangles: list[tuple[int, int, int]] = []

    for t in triangle_list:
        p1 = np.array([t[0], t[1]], dtype=np.float32)
        p2 = np.array([t[2], t[3]], dtype=np.float32)
        p3 = np.array([t[4], t[5]], dtype=np.float32)

        if not (0 <= p1[0] < width and 0 <= p1[1] < height):
            continue
        if not (0 <= p2[0] < width and 0 <= p2[1] < height):
            continue
        if not (0 <= p3[0] < width and 0 <= p3[1] < height):
            continue

        idx = []
        for p in (p1, p2, p3):
            d = np.linalg.norm(points - p, axis=1)
            idx.append(int(np.argmin(d)))

        i1, i2, i3 = idx
        if i1 != i2 and i2 != i3 and i1 != i3:
            tri = (i1, i2, i3)
            if tri not in triangles:
                triangles.append(tri)

    return triangles


def _warp_triangle(
    src: np.ndarray,
    dst: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
) -> None:
    r_src = cv2.boundingRect(t_src.astype(np.float32))
    r_dst = cv2.boundingRect(t_dst.astype(np.float32))

    t_src_rect = np.array([[t_src[i][0] - r_src[0], t_src[i][1] - r_src[1]] for i in range(3)], dtype=np.float32)
    t_dst_rect = np.array([[t_dst[i][0] - r_dst[0], t_dst[i][1] - r_dst[1]] for i in range(3)], dtype=np.float32)

    patch_src = src[r_src[1]:r_src[1] + r_src[3], r_src[0]:r_src[0] + r_src[2]]
    if patch_src.size == 0 or r_dst[2] <= 0 or r_dst[3] <= 0:
        return

    warp_mat = cv2.getAffineTransform(t_src_rect, t_dst_rect)
    patch_warp = cv2.warpAffine(
        patch_src,
        warp_mat,
        (r_dst[2], r_dst[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((r_dst[3], r_dst[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_dst_rect), (1.0, 1.0, 1.0), lineType=cv2.LINE_AA)

    dst_patch = dst[r_dst[1]:r_dst[1] + r_dst[3], r_dst[0]:r_dst[0] + r_dst[2]].astype(np.float32)
    dst_patch = dst_patch * (1.0 - mask) + patch_warp.astype(np.float32) * mask
    dst[r_dst[1]:r_dst[1] + r_dst[3], r_dst[0]:r_dst[0] + r_dst[2]] = np.clip(dst_patch, 0, 255).astype(np.uint8)


def _piecewise_affine_warp(image: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    triangles = _delaunay_triangles(src_pts, w, h)
    out = image.copy()

    for i1, i2, i3 in triangles:
        t_src = np.array([src_pts[i1], src_pts[i2], src_pts[i3]], dtype=np.float32)
        t_dst = np.array([dst_pts[i1], dst_pts[i2], dst_pts[i3]], dtype=np.float32)
        _warp_triangle(image, out, t_src, t_dst)

    return out


def _scale_points(points: np.ndarray, indices: list[int], scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-3:
        return points
    edited = points.copy()
    pts = edited[indices]
    center = pts.mean(axis=0)
    edited[indices] = center + (pts - center) * scale
    return edited


def _slim_points(points: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 1e-3:
        return points

    edited = points.copy()
    nose = edited[_NOSE_TIP]

    for idx in _JAW_LEFT:
        edited[idx] = edited[idx] + (nose - edited[idx]) * (0.28 * strength)
    for idx in _JAW_RIGHT:
        edited[idx] = edited[idx] + (nose - edited[idx]) * (0.28 * strength)

    # cheek major points more aggressive than jawline
    edited[234] = edited[234] + (nose - edited[234]) * (0.35 * strength)
    edited[454] = edited[454] + (nose - edited[454]) * (0.35 * strength)
    return edited


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

        h, w = bgr_image.shape[:2]
        src_pts = landmarks.points.copy()
        dst_pts = src_pts.copy()

        dst_pts = _scale_points(dst_pts, _MOUTH, mouth_scale)
        dst_pts = _scale_points(dst_pts, _LEFT_EYE, eye_scale)
        dst_pts = _scale_points(dst_pts, _RIGHT_EYE, eye_scale)
        dst_pts = _slim_points(dst_pts, slim_face_strength)

        border = _add_border_points(w, h)
        src_all = np.vstack([src_pts, border])
        dst_all = np.vstack([dst_pts, border])

        output = _piecewise_affine_warp(bgr_image, src_all, dst_all)

        mask = _hair_mask(output.shape, landmarks)
        output = _shift_hair_color(output, mask, hair_hue_shift, hair_saturation_boost)

        applied = (
            f"发色偏移={hair_hue_shift:.0f}, 发色饱和度={hair_saturation_boost:.2f}, "
            f"嘴巴={mouth_scale:.2f}, 眼睛={eye_scale:.2f}, 瘦脸={slim_face_strength:.2f}"
        )
        return output, f"编辑完成（液化级网格变形）。{applied}"
