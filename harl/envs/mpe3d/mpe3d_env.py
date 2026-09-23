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
import pyvista as pv
import cv2
import os
from collections import deque
try:
    from util import *
except ImportError:
    from .util import *


# ---------- Agents ----------
@dataclass
class Kinematics:
    max_speed: float

@dataclass
class Agent:
    pos: np.ndarray
    vel: np.ndarray
    kin: Kinematics
    radius: float
    planar: bool

    def step(self, v_cmd: np.ndarray, world: Continuous3DWorld, dt: float, margin: float):
        v = v_cmd.astype(np.float32).copy()
        if self.planar: v[2] = 0.0
        s = np.linalg.norm(v)
        if s > self.kin.max_speed: v *= (self.kin.max_speed / max(1e-6, s))
        new_pos = self.pos + v * dt
        if self.planar: new_pos[2] = 0.0
        new_pos = world.project_to_free(new_pos, self.radius + margin)
        self.vel = v; self.pos = new_pos

# ---------- Evader ----------
# --- Evader (replace your policy method) ---
class Evader:
    def __init__(self, pos: np.ndarray, speed: float = 1.6, radius: float = 0.4, planar: bool = True):
        self.pos = pos.astype(np.float32); self.vel = np.zeros(3,np.float32)
        self.speed=float(speed); self.radius=float(radius); self.planar=bool(planar)

    def policy(self, purs: np.ndarray, world: Continuous3DWorld) -> np.ndarray:
        if purs.size == 0:
            return np.zeros(3, np.float32)
        # vector from evader to pursuers
        d = purs - self.pos[None, :]                  # shape (N,3)
        i = int(np.argmin(np.linalg.norm(d[:, :2], axis=1)))  # nearest in XY
        away = -normalize_vec3(d[i], planar=self.planar)      # 3D unit vector
        grad = normalize_vec3(sdf_grad(world.sdf, self.pos), planar=self.planar)
        u = normalize_vec3(away + 0.3 * grad, planar=self.planar)
        return u * self.speed

    def step(self, purs: np.ndarray, world: Continuous3DWorld, dt: float):
        v = self.policy(purs, world)
        p = self.pos + v * dt
        if self.planar:
            p[2] = 0.0
        self.pos = world.project_to_free(p, self.radius + 0.3)
        self.vel = v

# ---------- Planner messages ----------
@dataclass
class GoalMsg:
    ugv_id: int
    pos_xy: Tuple[float,float]
    score: float
    eta: float
    gap_after: float
    choke_id: int
    stamp: int

# ---------- Env (UGV-only MARL; UAV = Coordinator; Stage-1 planner) ----------
class MPE3DContinuousEnv(gym.Env):
    metadata = {"render_modes": ["human","rgb_array"], "render_fps": 5}

    def __init__(
        self,
        map_size: Tuple[int,int,int]=(100,100,30),
        n_ugv: int = 6,
        n_evader: int = 1,
        render_mode: Optional[str] = None,
        seed: int = 0,
        physics_hz: int = 60,
        planner_hz: int = 2,
        uav_speed: float = 4.0,
        uav_follow_gain_xy: float = 2.0,   # how aggressively UAV recenters to centroid
        uav_follow_gain_z: float = 0.6,    # how altitude scales with spread
        uav_alt_min: float = 6.0,
        uav_alt_max: float = 16.0,
        uav_keepout_margin: float = 2.0,   # extra XY buffer beyond farthest UGV
        uav_obstacle_push: float = 1.0,    # strength of SDF-based obstacle push
        ugv_speed: float = 2.2,
        evader_speed: float = 1.6,
        agent_radius: float = 0.7,
        safety_margin: float = 0.3,
        bev_res: Tuple[int,int]=(64,64),
        bev_range_min: float = 30.0,
        bev_range_max: float = 80.0,
        bev_margin: float = 5.0,
        lidar_beams: int = 64,
        lidar_range: float = 30.0,
        max_steps: int = 1200,
        capture_radius: float = 1.0,
        capture_dwell: float = 0.6,
        # --- 3D video options ---
        record_video: bool = False,
        video_file: str = "./videos/mpe3d_run.mp4",
        voxel_z_stride: int = 1,
        voxel_obstacle_alpha: float = 0.30,
        trail_len: int = 60,
        # NEW explicit controls
        record_video_2d: bool | None = None,  # default: follow record_video
        record_video_3d: bool | None = None,  # default: follow record_video
    ):
        super().__init__()
        H,W,D = map_size
        self.world = Continuous3DWorld(WorldConfig(size=map_size, seed=seed, agent_radius=agent_radius, dt=1.0/physics_hz))
        self.rng = np.random.default_rng(seed)
        self.render_mode = render_mode

        self.physics_hz = int(physics_hz); self.dt = 1.0 / float(self.physics_hz)
        self.planner_hz = int(planner_hz); self.plan_interval = max(1, self.physics_hz // self.planner_hz)

        self.n_ugv = int(n_ugv); self.n_evader = int(n_evader)
        self.agent_radius = float(agent_radius); self.safety_margin = float(safety_margin)

        self.uav = Agent(pos=np.array([H*0.3, W*0.2, min(D-2, 10.0)], np.float32),
                         vel=np.zeros(3,np.float32), kin=Kinematics(max_speed=uav_speed),
                         radius=agent_radius, planar=False)
        
        self.uav.kin.max_speed = float(uav_speed)
        self.uav_follow_gain_xy = float(uav_follow_gain_xy)
        self.uav_follow_gain_z  = float(uav_follow_gain_z)
        self.uav_alt_min = float(uav_alt_min)
        self.uav_alt_max = float(uav_alt_max)
        self.uav_keepout_margin = float(uav_keepout_margin)
        self.uav_obstacle_push = float(uav_obstacle_push)

        self.ugvs: List[Agent] = []
        for i in range(self.n_ugv):
            y = H*0.8; x = W*(0.1 + 0.8*i/max(1,self.n_ugv-1))
            self.ugvs.append(Agent(pos=np.array([y,x,0.0],np.float32), vel=np.zeros(3,np.float32),
                                   kin=Kinematics(max_speed=ugv_speed), radius=agent_radius, planar=True))
        self.evaders: List[Evader] = [Evader(pos=np.array([H*0.45,W*0.7,0.0],np.float32), speed=evader_speed, planar=True)
                                      for _ in range(self.n_evader)]

        self.bev_res = bev_res
        self.bev_range_min = float(bev_range_min); self.bev_range_max = float(bev_range_max)
        self.bev_margin = float(bev_margin); self._bev_range_cur = float(bev_range_min)

        self.lidar_beams = int(lidar_beams); self.lidar_range = float(lidar_range)

        self.max_steps = int(max_steps); self.capture_radius=float(capture_radius); self.capture_dwell=float(capture_dwell)

        # Planner caches
        self._goals: List[GoalMsg] = []
        self._planner_meta = np.zeros(2, np.float32)  # [min_gap, avg_score]

        # Actions: UGV-only
        low = []; high=[]
        for _ in range(self.n_ugv):
            low += [-ugv_speed, -ugv_speed]
            high += [ugv_speed, ugv_speed]
        self.action_space = spaces.Box(low=np.array(low,np.float32), high=np.array(high,np.float32), dtype=np.float32)

        # Observations include goal_feats
        bev_shape = (self.bev_res[0], self.bev_res[1], 2)
        vec_dim = 6 + 5*self.n_ugv + 5*self.n_evader
        goal_shape = (self.n_ugv, 7)  # [dx,dy,dist,bearing,score,eta,gap_after]
        self.observation_space = spaces.Dict({
            "bev": spaces.Box(low=0, high=255, shape=bev_shape, dtype=np.uint8),
            "vec": spaces.Box(low=-np.inf, high=np.inf, shape=(vec_dim,), dtype=np.float32),
            "lidar0": spaces.Box(low=0.0, high=float(self.lidar_range), shape=(self.lidar_beams,), dtype=np.float32),
            "goal_feats": spaces.Box(low=-np.inf, high=np.inf, shape=goal_shape, dtype=np.float32),
            "planner_meta": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        })

        self._t = 0; self._dwell = 0.0

        # --- 3D video state ---
        # video paths & flags
        self.video3d_path = str(video_file)
        base, ext = os.path.splitext(self.video3d_path)
        self.video2d_path = base + "_topdown" + (ext or ".mp4")
        self.record_video_2d = bool(record_video if record_video_2d is None else record_video_2d)
        self.record_video_3d = bool(record_video if record_video_3d is None else record_video_3d)

        # writers
        self._video2d_writer = None
        self._video3d_writer = None
        self._video2d_frames = 0
        self._video3d_frames = 0

        # 3D render knobs
        self.voxel_z_stride = int(voxel_z_stride)
        self.voxel_obstacle_alpha = float(voxel_obstacle_alpha)

        # --- trajectory trails ---
        self.trail_len = int(trail_len)
        self._ugv_trails: list[deque] = []   # each: deque[(y,x,z)]

    # ---------- Reset ----------

    def _place_ugvs_bottom(self, min_spacing: float | int | None = None):
        """Place UGVs on bottom row, non-obstacle cells, with spacing and clearance."""
        H, W, D = self.world.env.map_H, self.world.env.map_W, self.world.env.map_D
        y = H - 1  # bottom row reserved free in GridEnv3D.set_obstacles
        margin = float(self.agent_radius + self.safety_margin)

        # candidate columns with sufficient clearance on ground z=0
        cand = []
        for x in range(W):
            if self.world.env.obstacles[y, x, 0] != 0:
                continue
            if float(self.world.env.dist_to_obs[y, x, 0]) < margin:
                continue
            cand.append(x)

        if not cand:
            # fallback: scan from bottom-2 upward
            for yy in range(H - 2, max(-1, H - 10), -1):
                for x in range(W):
                    if self.world.env.obstacles[yy, x, 0] == 0 and float(self.world.env.dist_to_obs[yy, x, 0]) >= margin:
                        cand.append((yy, x))
                if cand:
                    break

        # normalize candidate format to (y, x)
        if cand and isinstance(cand[0], int):
            cand = [(y, x) for x in cand]  # keep bottom row

        # spacing in cells; default ≈ 2*radius + margin + slack
        if min_spacing is None:
            min_spacing_cells = int(np.ceil(2.0 * self.agent_radius + self.safety_margin + 2.0))
        else:
            min_spacing_cells = int(np.ceil(float(min_spacing)))
        min_spacing_cells = max(2, min_spacing_cells)

        # greedy pick with relaxation if needed
        def try_pick(spacing_cells: int):
            picks = []
            for cy, cx in cand:
                ok = True
                for py, px in picks:
                    if np.hypot(cx - px, cy - py) < spacing_cells:
                        ok = False; break
                if ok:
                    picks.append((cy, cx))
                    if len(picks) == self.n_ugv:
                        break
            return picks

        picks = try_pick(min_spacing_cells)
        relax = min_spacing_cells
        while len(picks) < self.n_ugv and relax > 1:
            relax -= 1
            picks = try_pick(relax)

        # if still short, fill remaining uniformly across width at bottom with projection
        while len(picks) < self.n_ugv:
            frac = (len(picks) + 0.5) / float(self.n_ugv)
            cx = int(np.clip(round(frac * (W - 1)), 0, W - 1))
            candidate = (y, cx)
            # avoid duplicates
            if candidate not in picks:
                picks.append(candidate)
            else:
                # linear probe
                for dx in range(1, max(4, W // self.n_ugv)):
                    for sx in (cx - dx, cx + dx):
                        if 0 <= sx < W and (y, sx) not in picks:
                            picks.append((y, sx)); break
                    if len(picks) >= self.n_ugv:
                        break

        # commit positions with z=0 and project to be float-safe
        for i, (py, px) in enumerate(picks[:self.n_ugv]):
            start = np.array([float(py), float(px), 0.0], dtype=np.float32)
            pos = self.world.project_to_free(start, margin, iters=1)
            pos[2] = 0.0
            self.ugvs[i].pos = pos
            self.ugvs[i].vel[:] = 0.0

    # helper: does BEV centered at pos cover all UGVs?
    def _would_cover_all_ugvs(self, pos: np.ndarray) -> bool:
        if self.n_ugv == 0:
            return True
        slack = max(1.0, 0.5 * getattr(self, "uav_keepout_margin", 2.0))
        dists = [float(np.linalg.norm(pos[:2] - g.pos[:2])) for g in self.ugvs]
        return (max(dists) if dists else 0.0) <= (self._bev_range_cur - self.bev_margin - slack)

    def _init_uav_over_ugvs(self):
        H, W, D = self.world.env.map_H, self.world.env.map_W, self.world.env.map_D
        min_clear = self.uav.radius + self.safety_margin

        # centroid & spread
        if self.n_ugv == 0:
            cxy = np.array([H * 0.5, W * 0.5], np.float32); r_group = 0.0
        else:
            P = np.stack([g.pos[:2] for g in self.ugvs], axis=0)
            cxy = P.mean(axis=0)
            r_group = float(np.max(np.linalg.norm(P - cxy[None, :], axis=1))) if P.size else 0.0

        # altitude based on spread
        z_lo = getattr(self, "uav_alt_min", 6.0)
        z_hi = getattr(self, "uav_alt_max", 16.0)
        z_gain = getattr(self, "uav_follow_gain_z", 0.6)
        z_des = float(np.clip(z_lo + z_gain * r_group, z_lo, min(z_hi, D - 1.0)))

        # initial target: directly above centroid
        tgt = np.array([cxy[0], cxy[1], z_des], np.float32)

        # obstacle push if needed
        d_here = trilinear_sample(self.world.sdf, np.array([cxy[0], cxy[1], z_des], np.float32))
        if d_here < min_clear:
            g = sdf_grad(self.world.sdf, np.array([cxy[0], cxy[1], z_des], np.float32))
            tgt = tgt + g * (min_clear - d_here + 1e-3)

        # snap safe
        tgt = self.world.project_to_free(tgt, min_clear, iters=2)

        # commit & zero vel
        self.uav.pos = tgt.astype(np.float32)
        self.uav.vel[:] = 0.0

        # ensure BEV coverage with small corrections
        for _ in range(3):
            self._update_bev_range()
            if self._would_cover_all_ugvs(self.uav.pos):
                break
            # move 35% toward centroid and raise z a bit
            self.uav.pos[:2] = 0.65 * self.uav.pos[:2] + 0.35 * cxy
            self.uav.pos[2] = float(np.clip(self.uav.pos[2] + 0.25 * max(1.0, r_group),
                                            z_lo, min(z_hi, D - 1.0)))
            self.uav.pos = self.world.project_to_free(self.uav.pos, min_clear, iters=1)

    def _seed_trails_from_current(self):
        """Initialize/resize UGV trails to current positions."""
        self._ugv_trails = []
        for g in self.ugvs:
            dq = deque(maxlen=max(1, self.trail_len))
            # seed a few copies so short trails render immediately
            for _ in range(min(3, self.trail_len)):
                dq.append((float(g.pos[0]), float(g.pos[1]), float(getattr(g.pos, "__len__", lambda: 0) and g.pos[2] if g.pos.shape[0] >= 3 else 0.0)))
            self._ugv_trails.append(dq)

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.world = Continuous3DWorld(WorldConfig(size=self.world.cfg.size,
                                                seed=int(self.rng.integers(0, 1_000_000_000)),
                                                dt=self.dt,
                                                obstacle_density=self.world.cfg.obstacle_density,
                                                block=self.world.cfg.block,
                                                min_height=self.world.cfg.min_height,
                                                max_height=self.world.cfg.max_height))
        H, W, D = self.world.env.map_H, self.world.env.map_W, self.world.env.map_D

        # UGVs: bottom row, free cells, spaced
        if len(self.ugvs) != self.n_ugv:
            self.ugvs = [self.ugvs[i] if i < len(self.ugvs) else
                        type(self.ugvs[0])(pos=np.zeros(3, np.float32),
                                            vel=np.zeros(3, np.float32),
                                            kin=self.ugvs[0].kin,
                                            radius=self.ugvs[0].radius,
                                            planar=True)
                        for i in range(self.n_ugv)] if self.ugvs else [
                            Agent(pos=np.zeros(3, np.float32),
                                vel=np.zeros(3, np.float32),
                                kin=Kinematics(max_speed=2.2),
                                radius=self.agent_radius,
                                planar=True) for _ in range(self.n_ugv)
                        ]
        self._place_ugvs_bottom()

        # Evaders: near pursuers but not encircled
        self._place_evaders_near_pursuers(r_min=6.0, r_max=14.0, min_gap_deg=120.0, tries=64, edge_bias=0.35)

        # UAV: above UGV centroid; ensure BEV covers all UGVs
        self._init_uav_over_ugvs()

        # --- seed trails from current state ---
        self._seed_trails_from_current()

        # counters, first goals, video
        self._t = 0; self._dwell = 0.0
        self._update_bev_range()
        self._goals = self.planner_tick(stamp=self._t)

        self._close_video_writers()
        if cv2 is not None:
            # 2D always possible
            if self.record_video_2d:
                try:
                    f2d = self._render_topdown()
                    self._open_2d_writer(f2d)
                except Exception as e:
                    print(f"[2D] init failed: {e}")
            # 3D if requested
            if self.record_video_3d and pv is not None:
                try:
                    f3d = self._render_3d_voxels_frame(elev=25.0, azim=45.0)
                    self._open_3d_writer(f3d)
                except Exception as e:
                    print(f"[3D] init failed: {e} (will continue without 3D)")
        return self._get_obs(), {}


    def _angular_gap_ok(self, epos_xy: np.ndarray, purs_xy: np.ndarray, min_gap_rad: float) -> bool:
        """Return True if angles from epos to pursuers have a gap >= min_gap_rad."""
        if purs_xy.shape[0] < 2:
            return True  # cannot be encircled by <2 pursuers
        v = purs_xy - epos_xy[None, :]
        ang = np.arctan2(v[:, 0], v[:, 1])  # atan2(y,x) with our indexing
        ang = np.sort((ang + 2*np.pi) % (2*np.pi))
        gaps = np.diff(np.concatenate([ang, ang[:1] + 2*np.pi]))
        return bool(np.max(gaps) >= float(min_gap_rad))

    # 3) evader placement near pursuers but not encircled
    def _place_evaders_near_pursuers(
        self,
        r_min: float = 6.0,
        r_max: float = 14.0,
        min_gap_deg: float = 120.0,
        tries: int = 64,
        edge_bias: float = 0.35,
    ):
        """
        Spawn evaders on ground near UGV cluster (z=0), ensuring a large angular gap exists.
        - r in [r_min, r_max] around UGV centroid
        - keep clearance >= radius + margin
        - avoid being within capture_radius initially
        - project to free cells and clamp to map
        """
        H, W, D = self.world.env.map_H, self.world.env.map_W, self.world.env.map_D
        margin = float(self.agent_radius + self.safety_margin)
        min_gap_rad = float(np.deg2rad(min_gap_deg))
        P = np.stack([g.pos[:2] for g in self.ugvs], axis=0) if self.n_ugv > 0 else np.zeros((0, 2), np.float32)
        if P.shape[0] == 0:
            # fallback: center-right
            base = np.array([H * 0.5, W * 0.7], np.float32)
        else:
            # centroid, nudged toward interior (away from closest boundary) to avoid being out-of-bounds
            c = P.mean(axis=0)
            # inward bias: push slightly away from edges
            inward = np.array([
                (0.5 - np.clip(c[0] / max(H - 1, 1), 0, 1)),
                (0.5 - np.clip(c[1] / max(W - 1, 1), 0, 1)),
            ], np.float32)
            inward = normalize_vec(inward)
            base = c + inward * edge_bias * max(H, W) * 0.05

        rng = self.rng
        placed = 0
        for e in self.evaders:
            ok_pos = None
            # multi-try ring sampling around base
            for _ in range(max(1, int(tries))):
                r = float(rng.uniform(r_min, r_max))
                th = float(rng.uniform(0, 2 * np.pi))
                cand = np.array([base[0] + r * np.sin(th), base[1] + r * np.cos(th), 0.0], np.float32)
                # clamp & project to free ground
                cand = self.world.project_to_free(cand, margin, iters=1)
                cand[2] = 0.0
                # clear of obstacles?
                if trilinear_sample(self.world.sdf, cand) < margin:
                    continue
                # not too close to any pursuer (avoid immediate capture)
                if P.shape[0] > 0:
                    dmin = float(np.min(np.linalg.norm(P - cand[:2][None, :], axis=1)))
                    if dmin <= max(2.0 * self.capture_radius, 1.2 * self.agent_radius + self.safety_margin):
                        continue
                    # check angular gap
                    if not self._angular_gap_ok(cand[:2], P, min_gap_rad):
                        continue
                ok_pos = cand
                break

            if ok_pos is None:
                # fallback: place opposite to UGV centroid direction from map center
                center = np.array([H / 2.0, W / 2.0], np.float32)
                dir_out = normalize_vec((base - center))
                if np.allclose(dir_out, 0.0):
                    dir_out = np.array([1.0, 0.0], np.float32)
                cand = np.array([base[0] + r_max * dir_out[0],
                                 base[1] + r_max * dir_out[1], 0.0], np.float32)
                ok_pos = self.world.project_to_free(cand, margin, iters=2)
                ok_pos[2] = 0.0

            e.pos = ok_pos
            e.vel[:] = 0.0
            placed += 1

    # ---------- Step ----------
    def step(self, ugv_action: np.ndarray):
        self._t += 1

        # Planner tick (low rate)
        if (self._t % self.plan_interval) == 0 or len(self._goals) != self.n_ugv:
            self._goals = self.planner_tick(stamp=self._t)

        # Coordinator drives UAV
        v_uav = self._coordinator_tick()

        # Parse UGV actions
        a = np.asarray(ugv_action, np.float32).copy()
        assert a.shape == (2*self.n_ugv,), f"expected {(2*self.n_ugv,)}, got {a.shape}"
        v_ugv = a.reshape(self.n_ugv, 2)

        # Safety shield (UGVs)
        v_ugv = self._ugv_safety_shield(v_ugv)

        # Physics
        self.uav.step(v_uav, self.world, self.dt, self.safety_margin)
        for i,g in enumerate(self.ugvs):
            v = np.array([v_ugv[i,0], v_ugv[i,1], 0.0], np.float32)
            g.step(v, self.world, self.dt, self.safety_margin)
            # --- update UGV trail ---
            if i < len(self._ugv_trails):
                self._ugv_trails[i].append((float(g.pos[0]), float(g.pos[1]), float(g.pos[2])))


        # Evaders
        purs = np.stack([gg.pos for gg in self.ugvs] + [self.uav.pos], axis=0)
        for e in self.evaders: e.step(purs, self.world, self.dt)

        # Keep BEV coverage
        self._update_bev_range()

        # Append 3D frame
        if cv2 is not None:
            # 3D frame (guarded; don’t crash on headless)
            if self._video3d_writer is not None:
                try:
                    az = (self._video3d_frames * 2.0) % 360.0
                    f3d = self._render_3d_voxels_frame(elev=25.0, azim=float(az))
                    self._video3d_writer.write(self._bgr(f3d))
                    self._video3d_frames += 1
                except Exception as e:
                    print(f"[3D] write failed at t={self._t}: {e}. Disabling 3D writer.")
                    self._close_video3d()
            # 2D frame
            if self._video2d_writer is not None:
                try:
                    f2d = self._render_topdown()
                    self._video2d_writer.write(self._bgr(f2d))
                    self._video2d_frames += 1
                except Exception as e:
                    print(f"[2D] write failed at t={self._t}: {e}. Disabling 2D writer.")
                    self._close_video2d()

        # Reward/termination
        r, info = self._reward_and_info()
        terminated = info["success"]
        truncated  = (self._t >= self.max_steps) or info["severe_collision"]

        return self._get_obs(), float(r), bool(terminated), bool(truncated), info

    # ---------- Stage-1: Voronoi-like planner ----------
    def planner_tick(self, stamp: int) -> List[GoalMsg]:
        """
        Obstacle-aware ground heuristic: toward-evader + lateral gap-closure; conflict-spread.
        """
        goals: List[GoalMsg] = []
        sdf2 = self.world.sdf[:, :, 0]
        epos = np.mean(np.stack([e.pos for e in self.evaders], axis=0), axis=0)
        e_xy = epos[:2]

        P = np.stack([g.pos[:2] for g in self.ugvs], axis=0)
        if len(P) >= 2:
            vecs = normalize_vec(P - e_xy[None, :])  # shape (N,2), safe
            angs = np.arctan2(vecs[:, 0], vecs[:, 1])
            angs = np.sort((angs + 2*np.pi) % (2*np.pi))
            gaps = np.diff(np.concatenate([angs, angs[:1] + 2*np.pi]))
            max_gap = float(np.max(gaps))
        else:
            max_gap = 2*np.pi

        proposed_xy = []; scores = []
        H, W, _ = self.world.cfg.size
        for i, g in enumerate(self.ugvs):
            p = g.pos[:2]
            to_e = e_xy - p
            dist = float(np.linalg.norm(to_e))
            dir_pe = normalize_vec(to_e) if dist > 1e-6 else np.array([0.0, 0.0], np.float32)
            lat = np.array([-dir_pe[1], dir_pe[0]], np.float32)
            step_len = min(6.0, max(2.0, dist * 0.4))
            C = [
                p + dir_pe * step_len,
                p + lat * min(4.0, step_len * 0.7),
                p - lat * min(4.0, step_len * 0.7),
            ]
            c_keep, c_scores = [], []
            for c in C:
                c = np.array([np.clip(c[0], 0, H-1e-3), np.clip(c[1], 0, W-1e-3)], np.float32)
                nS = 12; lam = np.linspace(0, 1, nS)
                path_clear = np.min([
                    trilinear_sample(self.world.sdf, np.array([p[0]*(1-t)+c[0]*t, p[1]*(1-t)+c[1]*t, 0.0], np.float32))
                    for t in lam
                ])
                pred_gap = max(0.3, max_gap * (1.0 - 0.10))
                gain = (dist - np.linalg.norm(e_xy - c)) + 0.2 * (max_gap - pred_gap)
                score = 1.0 * gain + 0.3 * (path_clear)
                if path_clear >= (self.agent_radius + self.safety_margin):
                    c_keep.append(c); c_scores.append(score)
            if not c_keep:  # fallback: small step toward evader
                c_keep = [p + dir_pe * min(2.0, dist)]
                c_scores = [0.0]
            k = int(np.argmax(c_scores)); cxy = c_keep[k]; sc = float(c_scores[k])
            eta = float(np.linalg.norm(cxy - p) / max(1e-6, g.kin.max_speed))
            goals.append(GoalMsg(ugv_id=i, pos_xy=(float(cxy[0]), float(cxy[1])), score=sc, eta=eta,
                                 gap_after=max_gap, choke_id=-1, stamp=stamp))
            proposed_xy.append(cxy); scores.append(sc)

        # spread conflicting goals
        XY = np.stack(proposed_xy, axis=0)
        min_spacing = 1.5 * (self.agent_radius*2 + self.safety_margin)
        for _ in range(3):
            for i in range(len(XY)):
                for j in range(i+1, len(XY)):
                    d = np.linalg.norm(XY[i] - XY[j])
                    if d < min_spacing:
                        push = (min_spacing - d) * 0.5
                        dir_ij = normalize(XY[i] - XY[j])[0] if d > 1e-6 else np.array([1.0, 0.0], np.float32)
                        XY[i] += dir_ij * push; XY[j] -= dir_ij * push
                        XY[i] = np.array([np.clip(XY[i][0], 0, H-1e-3), np.clip(XY[i][1], 0, W-1e-3)], np.float32)
                        XY[j] = np.array([np.clip(XY[j][0], 0, H-1e-3), np.clip(XY[j][1], 0, W-1e-3)], np.float32)

        out: List[GoalMsg] = []
        for i, g in enumerate(goals):
            out.append(GoalMsg(ugv_id=g.ugv_id, pos_xy=(float(XY[i][0]), float(XY[i][1])),
                               score=g.score, eta=g.eta, gap_after=g.gap_after, choke_id=g.choke_id, stamp=g.stamp))

        self._planner_meta = np.array([max_gap, float(np.mean(scores)) if scores else 0.0], np.float32)
        return out

    # ---------- Goal features ----------
    def build_goal_feats(self) -> np.ndarray:
        feats = np.zeros((self.n_ugv, 7), np.float32)
        for g in self._goals:
            i = int(g.ugv_id)
            p = self.ugvs[i].pos[:2]
            dxy = np.array([g.pos_xy[0]-p[0], g.pos_xy[1]-p[1]], np.float32)
            dist = float(np.linalg.norm(dxy))
            bearing = float(np.arctan2(dxy[0], dxy[1]+1e-8))
            feats[i] = np.array([dxy[0], dxy[1], dist, bearing, g.score, g.eta, g.gap_after], np.float32)
        return feats

    # ---------- Coordinator (UAV, no learning) ----------
    def _coordinator_tick(self) -> np.ndarray:
        """
        Keep UAV near UGV centroid so BEV covers all UGVs, without orbiting.
        Adjust altitude with spread; push off obstacles via SDF; PD tracking.
        """
        H, W, D = self.world.env.map_H, self.world.env.map_W, self.world.env.map_D
        if self.n_ugv == 0:
            # idle hover near center at safe altitude
            tgt = np.array([H/2.0, W/2.0, max(self.uav_alt_min, min(self.uav_alt_max, self.uav.pos[2]))], np.float32)
            tgt = self.world.project_to_free(tgt, self.uav.radius + self.safety_margin)
            return self._pd_to(tgt, kp=2.0, kd=0.25, vmax=self.uav.kin.max_speed)

        # UGV centroid and spread
        P = np.stack([g.pos[:2] for g in self.ugvs], axis=0)  # (N,2)
        Cxy = P.mean(axis=0)
        r_group = float(np.max(np.linalg.norm(P - Cxy[None, :], axis=1))) if P.size else 0.0

        # Desired horizontal offset (small nudge away from obstacles)
        tgt_xy = Cxy.copy()
        # If UAV too far from centroid, pull it closer (keeps BEV efficient)
        err_xy = Cxy - self.uav.pos[:2]
        tgt_xy = self.uav.pos[:2] + self.uav_follow_gain_xy * 0.5 * err_xy * self.dt

        # Altitude: base on spread, clamped
        z_des = self.uav_alt_min + self.uav_follow_gain_z * r_group
        z_des = float(np.clip(z_des, self.uav_alt_min, min(self.uav_alt_max, D - 1.0)))

        # Provisional target and obstacle push
        tgt = np.array([tgt_xy[0], tgt_xy[1], z_des], np.float32)

        # Slightly push away from nearby obstacles (3D SDF)
        d_here = trilinear_sample(self.world.sdf, self.uav.pos)
        min_clear = self.uav.radius + self.safety_margin
        if d_here < min_clear:
            push = sdf_grad(self.world.sdf, self.uav.pos) * (self.uav_obstacle_push * (min_clear - d_here + 1e-3))
            tgt += push

        # Ensure target is safe (snaps to free & clamps to bounds)
        tgt = self.world.project_to_free(tgt, self.uav.radius + self.safety_margin, iters=1)

        # Guarantee BEV containment: if any UGV would fall out, pull target toward centroid
        if not self._would_cover_all_ugvs(tgt):
            # Move 30% toward centroid and raise a bit for larger footprint
            tgt[:2] = 0.7 * tgt[:2] + 0.3 * Cxy
            tgt[2] = float(np.clip(tgt[2] + 0.3 * r_group, self.uav_alt_min, min(self.uav_alt_max, D - 1.0)))
            tgt = self.world.project_to_free(tgt, self.uav.radius + self.safety_margin, iters=1)

        # PD control to target with speed clamp
        return self._pd_to(tgt, kp=2.0, kd=0.25, vmax=self.uav.kin.max_speed)

    # Helper: would the BEV centered at 'tgt' capture all UGVs?
    def _would_cover_all_ugvs(self, tgt_pos: np.ndarray) -> bool:
        if self.n_ugv == 0:
            return True
        r = self._bev_range_cur  # current range (will also be updated below)
        # Leave small slack so markers don't clip at edge
        slack = max(1.0, 0.5 * self.uav_keepout_margin)
        dists = [float(np.linalg.norm(tgt_pos[:2] - g.pos[:2])) for g in self.ugvs]
        return max(dists) <= (r - self.bev_margin - slack)

    # Helper: PD velocity toward target with clamp (kept minimal)
    def _pd_to(self, tgt: np.ndarray, kp: float, kd: float, vmax: float) -> np.ndarray:
        err = tgt - self.uav.pos
        v   = kp * err - kd * self.uav.vel
        s = float(np.linalg.norm(v))
        if s > vmax and s > 1e-6:
            v *= (vmax / s)
        return v.astype(np.float32)

    # Update BEV range so it comfortably includes all UGVs (no orbit)
    def _update_bev_range(self):
        if self.n_ugv == 0:
            self._bev_range_cur = self.bev_range_min
            return
        dists = [float(np.linalg.norm(self.uav.pos[:2] - g.pos[:2])) for g in self.ugvs]
        d_max = max(dists) if dists else 0.0
        # keepout buffer: ensure a margin beyond farthest UGV
        buffer = self.bev_margin + self.uav_keepout_margin
        r = d_max + buffer
        self._bev_range_cur = float(np.clip(r, self.bev_range_min, self.bev_range_max))

    # ---------- Safety shield (UGVs only) ----------
    def _ugv_safety_shield(self, v_ugv: np.ndarray) -> np.ndarray:
        out = v_ugv.copy()
        margin_obs = self.agent_radius + self.safety_margin
        for i,g in enumerate(self.ugvs):
            d = trilinear_sample(self.world.sdf, np.array([g.pos[0], g.pos[1], 0.0], np.float32))
            if d < margin_obs:
                out[i] += sdf_grad(self.world.sdf, np.array([g.pos[0], g.pos[1], 0.0], np.float32))[:2] * (g.kin.max_speed * (margin_obs - d + 1e-3))
        nn_margin = 2.0*self.agent_radius + self.safety_margin
        xy = [g.pos[:2] for g in self.ugvs]
        for i in range(self.n_ugv):
            for j in range(i+1, self.n_ugv):
                dij = xy[i] - xy[j]; L = np.linalg.norm(dij)
                if L < nn_margin:
                    dir_ = (dij / max(L, 1e-6)).astype(np.float32) if L > 1e-6 else np.array([1.0, 0.0], np.float32)
                    push = (nn_margin - L) * 0.5
                    out[i] += dir_ * push; out[j] -= dir_ * push
        for i in range(self.n_ugv):
            s = np.linalg.norm(out[i]); ms = self.ugvs[i].kin.max_speed
            if s > ms: out[i] *= (ms / s)
        return out.astype(np.float32)

    # ---------- Dynamic BEV ----------
    def _update_bev_range(self):
        if self.n_ugv == 0:
            self._bev_range_cur = self.bev_range_min; return
        dists = [float(np.linalg.norm(self.uav.pos[:2] - g.pos[:2])) for g in self.ugvs]
        d_max = max(dists) if dists else 0.0
        r = d_max + self.bev_margin
        self._bev_range_cur = float(np.clip(r, self.bev_range_min, self.bev_range_max))

    # ---------- Observations ----------
    def _get_obs(self) -> Dict[str,np.ndarray]:
        return {
            "bev": self._uav_bev(),
            "vec": self._vector_obs(),
            "lidar0": self._ugv_lidar(self.ugvs[0]) if self.n_ugv>0 else np.zeros(self.lidar_beams,np.float32),
            "goal_feats": self.build_goal_feats(),
            "planner_meta": self._planner_meta.copy(),
        }

    def _uav_bev(self) -> np.ndarray:
        H,W,D = self.world.cfg.size
        u = self.uav.pos; r = self._bev_range_cur
        ys = np.linspace(u[0]-r, u[0]+r, self.bev_res[0])
        xs = np.linspace(u[1]-r, u[1]+r, self.bev_res[1])
        img = np.zeros((self.bev_res[0], self.bev_res[1], 2), np.uint8)
        z_probe = max(0, min(D-1, int(u[2])))
        for iy,y in enumerate(ys):
            for ix,x in enumerate(xs):
                col = max(trilinear_sample(self.world.sdf, np.array([y,x,0.0],np.float32)),
                          trilinear_sample(self.world.sdf, np.array([y,x,float(z_probe)],np.float32)))
                img[iy,ix,0] = 255 if col >= (self.agent_radius+self.safety_margin) else 0
        def stamp(pos, val, rad):
            cy = int((pos[0]-(u[0]-r)) / (2*r) * (self.bev_res[0]-1))
            cx = int((pos[1]-(u[1]-r)) / (2*r) * (self.bev_res[1]-1))
            rr = max(1, int((rad / (2*r)) * self.bev_res[0] * 8))
            for dy in range(-rr, rr+1):
                for dx in range(-rr, rr+1):
                    yy, xx = cy+dy, cx+dx
                    if 0 <= yy < self.bev_res[0] and 0 <= xx < self.bev_res[1]:
                        if dy*dy+dx*dx <= rr*rr: img[yy,xx,1] = val
        for g in self.ugvs: stamp(g.pos, 180, g.radius)
        for e in self.evaders: stamp(e.pos, 80, e.radius)
        stamp(self.uav.pos, 220, self.uav.radius)
        return img

    def _ugv_lidar(self, ugv: Agent) -> np.ndarray:
        out = np.zeros(self.lidar_beams, np.float32)
        c = ugv.pos.copy(); c[2]=0.5
        for i in range(self.lidar_beams):
            ang = 2*math.pi*i/self.lidar_beams
            d = np.array([math.sin(ang), math.cos(ang), 0.0], np.float32)
            out[i] = raycast_sdf(self.world.sdf, c, d, self.lidar_range)
        return out

    def _vector_obs(self) -> np.ndarray:
        vec = [*self.uav.pos.tolist(), *self.uav.vel.tolist()]
        for g in self.ugvs:
            heading = math.atan2(g.vel[0]+1e-6, g.vel[1]+1e-6)
            d_obs = trilinear_sample(self.world.sdf, np.array([g.pos[0], g.pos[1], 0.0], np.float32))
            vec += [g.pos[0], g.pos[1], float(np.linalg.norm(g.vel[:2])), float(heading), float(d_obs)]
        for e in self.evaders:
            heading = math.atan2(e.vel[0]+1e-6, e.vel[1]+1e-6)
            d_obs = trilinear_sample(self.world.sdf, np.array([e.pos[0], e.pos[1], 0.0], np.float32))
            vec += [e.pos[0], e.pos[1], float(np.linalg.norm(e.vel[:2])), float(heading), float(d_obs)]
        return np.asarray(vec, np.float32)

    # ---------- Reward / Termination ----------
    def _capture_logic(self) -> Tuple[bool,float]:
        purs = np.stack([g.pos for g in self.ugvs] + [self.uav.pos], axis=0)
        min_prox = np.inf
        for e in self.evaders:
            d = np.min(np.linalg.norm(purs[:,:2] - e.pos[None,:2], axis=1))
            min_prox = min(min_prox, d)
        if min_prox <= self.capture_radius: self._dwell += self.dt
        else: self._dwell = 0.0
        return (self._dwell >= self.capture_dwell), float(min_prox)

    def _collision_penalty(self) -> Tuple[float,bool]:
        agents = [self.uav] + self.ugvs
        pen = 0.0; severe = False
        for i in range(len(agents)):
            for j in range(i+1,len(agents)):
                d = np.linalg.norm(agents[i].pos - agents[j].pos)
                th = (agents[i].radius + agents[j].radius)*0.8
                if d < th:
                    pen += (th - d)
                    if d < th*0.5: severe = True
        return float(pen), bool(severe)

    def _goal_progress(self) -> float:
        s = 0.0
        for g in self._goals:
            i = int(g.ugv_id); p = self.ugvs[i].pos[:2]; tgt = np.array([g.pos_xy[0], g.pos_xy[1]], np.float32)
            s += -float(np.linalg.norm(tgt - p))
        return s / max(1, len(self._goals))

    def _reward_and_info(self):
        success, min_prox = self._capture_logic()
        col_pen, severe = self._collision_penalty()
        purs = np.stack([gg.pos for gg in self.ugvs] + [self.uav.pos], axis=0)
        epos = np.mean(np.stack([e.pos for e in self.evaders], axis=0), axis=0)
        mean_dist = float(np.mean(np.linalg.norm(purs[:,:2] - epos[None,:2], axis=1)))
        min_gap = float(self._planner_meta[0])
        goal_prog = self._goal_progress()

        r = 0.0
        r += 1.0 * (1.0 / (mean_dist + 1.0))
        r += 0.25 * (-min_gap / (2*np.pi))
        r += 0.30 * (-goal_prog) * 0.05
        r -= 0.02
        r -= 1.0 * col_pen
        r += (10.0 if success else 0.0)

        dists = [float(np.linalg.norm(self.uav.pos[:2] - g.pos[:2])) for g in self.ugvs]
        max_in_bev = max(dists) <= (self._bev_range_cur - self.bev_margin + 1e-6)

        info = {
            "success": bool(success),
            "min_prox": float(min_prox),
            "mean_dist": float(mean_dist),
            "min_gap": float(min_gap),
            "goal_progress": float(goal_prog),
            "collision_pen": float(col_pen),
            "severe_collision": bool(severe),
            "bev_range_cur": float(self._bev_range_cur),
            "ugv_all_in_bev": bool(max_in_bev),
            "t": int(self._t),
        }
        return float(r), info

    # ---------- Render ----------
    def render(self):
        frame = self._render_topdown()
        if self.render_mode == "human":
            plt.imshow(frame); plt.axis("off"); plt.show(block=False); plt.pause(0.001)
        return frame

    def _render_topdown(self):
        H, W, D = self.world.env.map_H, self.world.env.map_W, self.world.env.map_D
        img = np.zeros((H,W,3), dtype=float)
        img[self.world.obstacles[:,:,0]==0] = [0.92,0.92,0.92]
        img[self.world.obstacles[:,:,0]==1] = [0.06,0.06,0.06]
        fig,ax = plt.subplots(figsize=(10,7), dpi=120)
        ax.imshow(img, origin="upper")

        # (optional) building footprints if you have them; else skip

        # --- UGV trails ---
        # Newer segments appear stronger (simple alpha ramp by segment age)
        for i, trail in enumerate(self._ugv_trails):
            if len(trail) < 2:
                continue
            # convert to numpy for faster slicing
            T = np.array(trail, dtype=np.float32)  # (L,3) [y,x,z]
            # draw segment-by-segment so we can fade with age
            L = T.shape[0]
            for k in range(L-1):
                y0,x0,_ = T[k]
                y1,x1,_ = T[k+1]
                age = (k+1) / float(L)  # 0..1
                lw = 1.0 + 2.0 * (1.0 - age)  # thicker when newer
                alpha = 0.25 + 0.55 * (1.0 - age)
                ax.plot([x0,x1], [y0,y1], '-', color='tab:blue', lw=lw, alpha=alpha)

        # --- agents (draw on top of trails) ---
        def draw(p, color, m='o', label=None):
            ax.plot(p[1], p[0], m, color=color, ms=6, mec="k", mew=0.6)
            if label:
                ax.text(p[1]+0.6, p[0]-0.6, label, fontsize=7, color="white",
                        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))
        for i,g in enumerate(self.ugvs):
            draw(g.pos, "tab:blue", 's', f"UGV{i}")
        draw(self.uav.pos, "tab:green", 'o', f"UAV z={int(self.uav.pos[2])}")
        for j,e in enumerate(self.evaders):
            draw(e.pos, "tab:red", 'o', f"E{j}")

        # --- BEV circle ---
        circ = plt.Circle((self.uav.pos[1], self.uav.pos[0]), self._bev_range_cur,
                        color="green", fill=False, ls="--", lw=1.2)
        ax.add_patch(circ)

        ax.set_xlim(-0.5,W-0.5); ax.set_ylim(H-0.5,-0.5); ax.axis("off")
        fig.canvas.draw()
        s,(w,h)=fig.canvas.tostring_rgb(), fig.canvas.get_width_height()
        frame = np.frombuffer(s, dtype=np.uint8).reshape(h,w,3)
        plt.close(fig)
        return frame

    # ---------- PyVista 3D: grid + frame ----------
    def _build_pyvista_voxel_grid(self, z_stride: int):
        """Create a PyVista grid + thresholded voxel mesh from self.world.obstacles. WHY: efficient off-screen render."""
        assert pv is not None, "pyvista not available"
        if z_stride is None or z_stride <= 0:
            z_stride = 1
        obs = self.world.obstacles[:, :, ::z_stride].astype(bool)  # (H, W, Dz)
        H, W, Dz = obs.shape
        grid = pv.UniformGrid() if hasattr(pv, "UniformGrid") else pv.ImageData()
        grid.dimensions = (W + 1, H + 1, Dz + 1)
        grid.spacing = (1.0, 1.0, float(z_stride))  # keep Z scale
        grid.origin = (0.0, 0.0, 0.0)
        scalars = obs.transpose(1, 0, 2).ravel(order="F").astype(np.uint8)
        grid.cell_data["occ"] = scalars
        voxels = grid.threshold(0.5, scalars="occ")
        return grid, voxels

    def _render_3d_voxels_frame(
        self,
        elev: float = 25.0,
        azim: float = 45.0,
        z_stride: int = None,
        obstacle_alpha: float = None,
    ):
        assert pv is not None, "pyvista not available"
        H, W, D = self.world.env.map_H, self.world.env.map_W, self.world.env.map_D
        z_stride = self.voxel_z_stride if z_stride is None else z_stride
        obstacle_alpha = self.voxel_obstacle_alpha if obstacle_alpha is None else obstacle_alpha
        if z_stride is None or z_stride <= 0:
            z_stride = max(1, int(self.voxel_z_stride))

        _, voxels = self._build_pyvista_voxel_grid(z_stride)

        plotter = pv.Plotter(off_screen=True, window_size=(1000, 700))
        plotter.set_background([214, 192, 168], top="white")

        # obstacles
        if voxels.n_cells > 0:
            plotter.add_mesh(voxels, color="black", opacity=float(obstacle_alpha), show_edges=False)

        # --- UGV trails as 3D polylines (z=agent z) ---
        try:
            for trail in self._ugv_trails:
                if len(trail) < 2:
                    continue
                P = np.array(trail, dtype=np.float32)  # (L,3) y,x,z
                # convert to XYZ for PyVista (x,y,z)
                XYZ = np.stack([P[:,1], P[:,0], P[:,2]], axis=1)
                # add as a single polyline
                # Create cells array: [n_points, 0,1,2,...]
                npts = XYZ.shape[0]
                cells = np.hstack(([npts], np.arange(npts, dtype=np.int32))).astype(np.int32)
                pline = pv.PolyData(XYZ)
                pline.lines = cells
                plotter.add_mesh(pline, color="royalblue", line_width=3, opacity=0.8)
        except Exception:
            # If polylines fail in headless, it's okay—skip trails in 3D
            pass

        # agents
        def _add_sphere(pos, color, r=0.8):
            y, x, z = pos
            sphere = pv.Sphere(center=(float(x), float(y), float(z)), radius=r)
            plotter.add_mesh(sphere, color=color, specular=0.2, smooth_shading=True)

        for g in self.ugvs: _add_sphere(g.pos, "royalblue", 0.9)
        _add_sphere(self.uav.pos, "seagreen", 1.0)
        for e in self.evaders: _add_sphere(e.pos, "crimson", 0.9)

        plotter.show_grid(color="black", font_size=8, location="outer")
        plotter.add_axes(line_width=2, labels_off=False)

        plotter.reset_camera()
        cam = plotter.camera
        cam.Elevation(float(elev))
        cam.Azimuth(float(azim))
        cam.Zoom(0.9)

        img = plotter.screenshot(return_img=True)
        plotter.close()
        frame = np.ascontiguousarray(img[..., :3].astype(np.uint8))
        return frame

    def _bgr(self, rgb):
        import cv2 as _cv2
        return _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)

    def _open_2d_writer(self, first_frame):
        if cv2 is None or self._video2d_writer is not None:
            return
        os.makedirs(os.path.dirname(self.video2d_path) or ".", exist_ok=True)
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = float(self.metadata.get("render_fps", 5) or 5)
        w2d = cv2.VideoWriter(self.video2d_path, fourcc, fps, (w, h))
        if w2d is not None and w2d.isOpened():
            w2d.write(self._bgr(first_frame))
            self._video2d_writer = w2d
            self._video2d_frames = 1
        else:
            print(f"[2D] Could not open writer at {self.video2d_path}")

    def _open_3d_writer(self, first_frame):
        if cv2 is None or self._video3d_writer is not None:
            return
        os.makedirs(os.path.dirname(self.video3d_path) or ".", exist_ok=True)
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = float(self.metadata.get("render_fps", 5) or 5)
        w3d = cv2.VideoWriter(self.video3d_path, fourcc, fps, (w, h))
        if w3d is not None and w3d.isOpened():
            w3d.write(self._bgr(first_frame))
            self._video3d_writer = w3d
            self._video3d_frames = 1
        else:
            print(f"[3D] Could not open writer at {self.video3d_path}")

    def _close_video2d(self):
        if self._video2d_writer is not None:
            try:
                self._video2d_writer.release()
                print(f"[2D] Saved video to: {self.video2d_path} (frames={self._video2d_frames})")
            finally:
                self._video2d_writer = None
                self._video2d_frames = 0

    def _close_video3d(self):
        if self._video3d_writer is not None:
            try:
                self._video3d_writer.release()
                print(f"[3D] Saved video to: {self.video3d_path} (frames={self._video3d_frames})")
            finally:
                self._video3d_writer = None
                self._video3d_frames = 0

    def _close_video_writers(self):
        self._close_video2d()
        self._close_video3d()

