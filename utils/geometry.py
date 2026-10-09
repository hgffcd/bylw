# -*- coding: utf-8 -*-
"""几何工具: 角度处理 / 线段-圆相交(遮挡判定) / 围捕角缺口."""
import math

import numpy as np


def wrap_angle(a):
    """把角度约束到 (-pi, pi]."""
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def segment_circle_intersect(p1, p2, center, radius) -> bool:
    """线段 p1-p2 是否与圆(center, radius)相交 —— 视线遮挡判定(式5-5第三条件)."""
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    center = np.asarray(center, float)
    d = p2 - p1
    f = p1 - center
    a = float(d @ d)
    if a < 1e-12:
        return float(np.linalg.norm(f)) <= radius
    t = -float(f @ d) / a
    t = min(max(t, 0.0), 1.0)
    nearest = p1 + t * d
    return float(np.linalg.norm(nearest - center)) <= radius


def sorted_gap_max(points, center) -> float:
    """以 center 为圆心, 计算各 point 方位角排序后的最大相邻夹角(围捕角最大缺口, 式5-7)."""
    points = np.asarray(points, float)
    if len(points) <= 1:
        return 2 * math.pi
    ang = np.sort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))
    gaps = np.diff(ang)
    gaps = np.append(gaps, 2 * math.pi - (ang[-1] - ang[0]))
    return float(gaps.max())
