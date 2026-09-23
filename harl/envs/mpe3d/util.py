from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def clamp(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(v, lo), hi)

def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def normalize_vec(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Safe normalization: 1D->1D, ND->normalize along last axis."""
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        return v / max(n, eps)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def normalize_vec3(v: np.ndarray, planar: bool = False, eps: float = 1e-8) -> np.ndarray:
    """Normalize to 3D vector; if planar, zero-out z."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)  # 1D
    if v.size == 2:
        v = np.array([v[0], v[1], 0.0], dtype=np.float32)
    elif v.size >= 3:
        v = np.array([v[0], v[1], v[2]], dtype=np.float32)
    n = float(np.linalg.norm(v))
    v = v / max(n, eps)
    if planar:
        v[2] = 0.0
    return v


def trilinear_sample(grid: np.ndarray, p: np.ndarray) -> float:
    """WHY: smooth SDF queries for safety shield."""
    H, W, D = grid.shape
    y, x, z = p
    y0 = int(np.floor(y)); x0 = int(np.floor(x)); z0 = int(np.floor(z))
    y1 = y0 + 1; x1 = x0 + 1; z1 = z0 + 1
    if (y0 < 0 or x0 < 0 or z0 < 0 or y1 >= H or x1 >= W or z1 >= D):
        return -1.0
    dy = y - y0; dx = x - x0; dz = z - z0
    c000 = grid[y0, x0, z0]; c100 = grid[y1, x0, z0]; c010 = grid[y0, x1, z0]; c110 = grid[y1, x1, z0]
    c001 = grid[y0, x0, z1]; c101 = grid[y1, x0, z1]; c011 = grid[y0, x1, z1]; c111 = grid[y1, x1, z1]
    c00 = c000 * (1 - dy) + c100 * dy
    c01 = c001 * (1 - dy) + c101 * dy
    c10 = c010 * (1 - dy) + c110 * dy
    c11 = c011 * (1 - dy) + c111 * dy
    c0 = c00 * (1 - dx) + c10 * dx
    c1 = c01 * (1 - dx) + c11 * dx
    return float(c0 * (1 - dz) + c1 * dz)

def sdf_grad(sdf: np.ndarray, p: np.ndarray, h: float = 1.0) -> np.ndarray:
    """WHY: push away from obstacles."""
    gy = trilinear_sample(sdf, p + np.array([ h,0,0])) - trilinear_sample(sdf, p + np.array([-h,0,0]))
    gx = trilinear_sample(sdf, p + np.array([0, h,0])) - trilinear_sample(sdf, p + np.array([0,-h,0]))
    gz = trilinear_sample(sdf, p + np.array([0,0, h])) - trilinear_sample(sdf, p + np.array([0,0,-h]))
    g = np.array([gy, gx, gz], np.float32)
    n = np.linalg.norm(g)
    return g / max(n, 1e-6)

def raycast_sdf(sdf: np.ndarray, start: np.ndarray, dir_unit: np.ndarray, max_range: float, step_scale: float = 0.9) -> float:
    """SDF sphere tracing for LiDAR."""
    p = start.astype(np.float32).copy()
    total = 0.0
    for _ in range(128):
        if total >= max_range:
            return max_range
        d = trilinear_sample(sdf, p)
        if d <= 0.5:
            return total
        step = max(0.1, float(d) * step_scale)
        p += dir_unit * step
        total += step
    return max_range


class GridEnv3D:
    def __init__(self, H: int, W: int, D: int):
        self.map_H, self.map_W, self.map_D = H, W, D
        self.known = np.zeros((H, W, D), dtype=np.uint8)
        self.obstacles = np.zeros((H, W, D), dtype=np.uint8)
        self.dist_to_obs = np.zeros((H, W, D), dtype=np.float32)
        self.hard_margin = 0.0

    def in_bounds(self, p: Tuple[int,int,int]) -> bool:
        return (0 <= p[0] < self.map_H and 0 <= p[1] < self.map_W and 0 <= p[2] < self.map_D)

    def passable(self, p: Tuple[int,int,int]) -> bool:
        return self.in_bounds(p) and self.obstacles[p] == 0

    def set_known(self, p: Tuple[int,int,int]):
        if self.in_bounds(p):
            self.known[p] = 1

    def set_obstacles(
        self,
        obstacle_prob: float = 0.06,
        seed: int = 0,
        block: int = 3,
        min_height: int = 2,
        max_height: int | None = None,
    ):
        rng = np.random.default_rng(seed)
        H, W, D = self.map_H, self.map_W, self.map_D
        if max_height is None:
            max_height = D

        ch = (H + block - 1) // block
        cw = (W + block - 1) // block
        coarse2d = (rng.random((ch, cw)) < obstacle_prob).astype(np.uint8)

        footprint = np.repeat(np.repeat(coarse2d, block, axis=0), block, axis=1)
        footprint = footprint[:H, :W].astype(np.uint8)
        footprint[-1, :] = 0  # free spawn strip at bottom row

        obs = np.zeros((H, W, D), dtype=np.uint8)
        ys, xs = np.where(footprint == 1)
        for y, x in zip(ys, xs):
            h = int(rng.integers(min_height, max_height + 1))
            z_top = min(D - 1, h - 1)
            obs[y, x, 0 : z_top + 1] = 1

        if D >= 3:
            obs[:, :, -1] = 0  # keep a free sky band

        self.obstacles = obs

    def update_distance_map(self):
        free = (self.obstacles == 0).astype(np.uint8)
        self.dist_to_obs = distance_transform_edt(free).astype(np.float32)

    def is_safe(self, p: Tuple[int,int,int], margin: float) -> bool:
        if not self.in_bounds(p) or self.obstacles[p] == 1:
            return False
        return float(self.dist_to_obs[p]) >= float(margin)

    def safest_neighbor(self, cur: Tuple[int,int,int]) -> Tuple[int,int,int]:
        ry, rx, rz = cur
        best, bestd = cur, -1.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    q = (ry + dy, rx + dx, rz + dz)
                    if not self.in_bounds(q) or self.obstacles[q] == 1:
                        continue
                    d = float(self.dist_to_obs[q])
                    if d > bestd:
                        bestd, best = d, q
        return best

    def project_to_safe(self, p: Tuple[int,int,int], margin: float, max_radius: int = 6) -> Tuple[int,int,int]:
        p = (int(p[0]), int(p[1]), int(p[2]))
        if self.is_safe(p, margin):
            return p
        H, W, D = self.map_H, self.map_W, self.map_D
        best, bestd = None, -1.0
        y0, x0, z0 = p
        for r in range(1, max_radius + 1):
            y_min, y_max = max(0, y0 - r), min(H - 1, y0 + r)
            x_min, x_max = max(0, x0 - r), min(W - 1, x0 + r)
            z_min, z_max = max(0, z0 - r), min(D - 1, z0 + r)
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    for z in range(z_min, z_max + 1):
                        q = (y, x, z)
                        if self.obstacles[q] == 1:
                            continue
                        d = float(self.dist_to_obs[q])
                        if d >= margin and d > bestd:
                            bestd, best = d, q
            if best is not None:
                return best
        return self.safest_neighbor(p)

# ---------- Continuous3DWorld (refactored to use GridEnv3D) ----------
@dataclass
class WorldConfig:
    size: Tuple[int,int,int] = (100,100,30)
    obstacle_density: float = 0.06  # maps to obstacle_prob
    seed: int = 0
    agent_radius: float = 0.7
    dt: float = 1/60
    block: int = 3
    min_height: int = 2
    max_height: int | None = None

class Continuous3DWorld:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        H, W, D = cfg.size
        self.env = GridEnv3D(H, W, D)
        # grounded 3D obstacles as requested
        self.env.set_obstacles(
            obstacle_prob=cfg.obstacle_density,
            seed=cfg.seed,
            block=cfg.block,
            min_height=cfg.min_height,
            max_height=cfg.max_height,
        )
        self.env.update_distance_map()
        # aliases used by the rest of the codebase
        self.obstacles = self.env.obstacles
        self.sdf = self.env.dist_to_obs

    def update_sdf(self):
        self.env.update_distance_map()
        self.sdf = self.env.dist_to_obs  # keep alias fresh

    def set_obstacles(self, obstacle_prob: float, seed: int, block: int = 3, min_height: int = 2, max_height: int | None = None):
        self.env.set_obstacles(obstacle_prob=obstacle_prob, seed=seed, block=block, min_height=min_height, max_height=max_height)
        self.update_sdf()

    def project_to_free(self, p: np.ndarray, margin: float, iters: int = 2) -> np.ndarray:
        """
        Float-safe projection to free space with requested margin.
        1) If current SDF(tri) >= margin -> clamp and return.
        2) Else snap to nearest cell and use discrete project_to_safe.
        3) Optionally nudge away from obstacles via small SDF gradient steps.
        """
        p = np.asarray(p, np.float32)
        H, W, D = self.env.map_H, self.env.map_W, self.env.map_D
        p = clamp(p, np.array([0,0,0], np.float32), np.array([H-1e-3, W-1e-3, D-1e-3], np.float32))

        if trilinear_sample(self.sdf, p) >= float(margin):
            return p

        # discrete fallback
        cell = tuple(np.clip(np.round(p).astype(int), [0,0,0], [H-1, W-1, D-1]))
        safe_cell = self.env.project_to_safe(cell, margin, max_radius=6)
        q = np.array([float(safe_cell[0]), float(safe_cell[1]), float(safe_cell[2])], np.float32)

        # gentle gradient ascent to increase clearance
        for _ in range(max(0, int(iters))):
            g = sdf_grad(self.sdf, q, h=1.0)
            if np.allclose(g, 0.0):
                break
            q = q + g * 0.5  # small step
            q = clamp(q, np.array([0,0,0], np.float32), np.array([H-1e-3, W-1e-3, D-1e-3], np.float32))
            if trilinear_sample(self.sdf, q) >= float(margin):
                break

        return q