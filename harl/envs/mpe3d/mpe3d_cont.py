# file: harl/envs/mpe3d/mpe3d_cont.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

# adjust this import to your actual path
from harl.envs.mpe3d.mpe3d_env import MPE3DContinuousEnv


def _as_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    return vars(x) if hasattr(x, "__dict__") else {}


class MPE3DContHARL:
    """
    HARL wrapper for MPE-3D (UGV-only, continuous actions).
    - Accepts per-agent list OR dict with stacked arrays.
    - Builds flat per-agent vectors; appends summarized globals (e.g., BEV).
    - Locks sizes at init; pads/truncates at runtime.
    - Actions: (n_agents,2) -> flattened to (2*n_agents,) for env.
    """

    def __init__(self, args: Union[Dict[str, Any], Any]):
        raw = _as_dict(args)
        cfg = raw.get("env_args", raw) or {}

        # encoding knobs
        self.include_bev: bool = bool(cfg.get("include_bev", False))
        self.bev_summary: str = str(cfg.get("bev_summary", "stats"))  # "stats" | "flatten"
        self.max_flatten: int = int(cfg.get("max_flatten", 4096))
        self.large_as_stats: bool = bool(cfg.get("large_as_stats", True))
        self.clip_obs_to_box: bool = bool(cfg.get("clip_obs_to_box", True))
        self.low_clip: float = float(cfg.get("low_clip", -10.0))
        self.high_clip: float = float(cfg.get("high_clip", 10.0))

        # env args
        self.n_ugv: int = int(cfg.get("n_ugv", 4))
        self.n_evader: int = int(cfg.get("n_evader", cfg.get("n_evaders", 2)))
        self.map_size: Tuple[int, int, int] = tuple(cfg.get("map_size", [100, 100, 30]))
        self.max_steps: int = int(cfg.get("episode_limit", cfg.get("max_steps", 1000)))
        self.action_clip: float = float(cfg.get("action_clip", 1.0))
        self.reward_scale: float = float(cfg.get("reward_scale", 1.0))
        self.record_video_2d: bool = bool(cfg.get("record_video_2d", False))
        self.record_video_3d: bool = bool(cfg.get("record_video_3d", False))
        self.video_file: str = str(cfg.get("video_file", "./videos/mpe3d_run.mp4"))

        # construct env
        self.env = MPE3DContinuousEnv(
            n_ugv=self.n_ugv,
            n_evader=self.n_evader,
            map_size=self.map_size,
            max_steps=self.max_steps,
            record_video=False,
            record_video_2d=self.record_video_2d,
            record_video_3d=self.record_video_3d,
            video_file=self.video_file,
        )
        self.n_agents: int = self.n_ugv

        # probe once, lock sizes
        obs0_raw, _ = self.env.reset(seed=0)
        obs0_list = self._to_per_agent_list(obs0_raw)               # <-- robust split
        enc0 = [self._encode_obs_probe(o) for o in obs0_list]
        assert all(e.ndim == 1 and e.size > 0 for e in enc0), "probe obs empty"
        self.obs_size: int = int(enc0[0].size)

        g0 = self._encode_global_raw()
        assert g0.ndim == 1 and g0.size > 0, "probe global empty"
        self.gvec_size: int = int(g0.size)

        self.share_obs_size: int = int(self.n_agents * self.obs_size + self.gvec_size)

        # (optional) for custom models
        self.vec_dim: int = self.obs_size
        self.bev_shape: Tuple[int, int, int] | None = None

        # spaces
        self.observation_space: List[gym.Space] = [
            Box(low=self.low_clip, high=self.high_clip, shape=(self.obs_size,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.share_observation_space: List[gym.Space] = [
            Box(low=self.low_clip, high=self.high_clip, shape=(self.share_obs_size,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self.action_space: List[gym.Space] = [
            Box(low=-self.action_clip, high=self.action_clip, shape=(2,), dtype=np.float32)
            for _ in range(self.n_agents)
        ]
        self._step_count = 0

    # ------------ HARL API ------------
    def reset(self):
        raw, _ = self.env.reset()
        self._step_count = 0
        pa = self._to_per_agent_list(raw)
        obs = [self._clip(self._fix_size(self._encode_obs_probe(o), self.obs_size)) for o in pa]
        state = self._build_state(obs)
        share = [state.copy() for _ in range(self.n_agents)]
        # IMPORTANT: return exactly 3 values for ShareVecEnv
        return obs, share, None

    def step(self, actions):
        a = np.asarray(actions, dtype=np.float32)
        a_flat = a.ravel() if a.ndim == 2 else a.reshape(-1)
        assert a_flat.size == 2 * self.n_agents, f"expected {2*self.n_agents}, got {a_flat.size}"

        raw_obs, r, terminated, truncated, env_info = self.env.step(a_flat)
        self._step_count += 1

        pa = self._to_per_agent_list(raw_obs)
        obs = [self._clip(self._fix_size(self._encode_obs_probe(o), self.obs_size)) for o in pa]
        state = self._build_state(obs)
        share = [state.copy() for _ in range(self.n_agents)]

        r_scalar = float(self.reward_scale * r)
        rewards = [np.array([r_scalar], dtype=np.float32) for _ in range(self.n_agents)]
        done = bool(terminated or truncated)
        dones = [done for _ in range(self.n_agents)]

        # Normalize info to List[dict] (runner expects list)
        if isinstance(env_info, list):
            infos = [dict(x) if isinstance(x, dict) else {} for x in env_info]
            if len(infos) != self.n_agents:
                infos = (infos + [{}] * self.n_agents)[: self.n_agents]
        elif isinstance(env_info, dict):
            infos = [dict(env_info) for _ in range(self.n_agents)]
        else:
            infos = [dict() for _ in range(self.n_agents)]

        # Return 6-tuple for off-policy runner
        return obs, share, rewards, dones, infos, None

    def seed(self, seed: int):
        np.random.seed(seed)

    def render(self):
        try:
            self.env.render()
        except Exception:
            pass

    def close(self):
        self.env.close()

    # ------------ Per-agent extraction ------------
    def _to_per_agent_list(self, obs_like: Any) -> List[Any]:
        """
        Normalize env outputs to a per-agent list.
        Supports:
        - list/tuple: returned directly.
        - dict: detect stacked per-agent arrays by leading dim == n_agents,
                and build agent vectors by concatenating those slices + summarized globals.
        """
        # Already list/tuple
        if isinstance(obs_like, (list, tuple)):
            assert len(obs_like) == self.n_agents, f"len={len(obs_like)} != n_agents={self.n_agents}"
            return list(obs_like)

        # Dict path
        if isinstance(obs_like, dict):
            d = obs_like

            # Identify stacked per-agent arrays (first dim == n_agents)
            per_agent_keys: List[str] = []
            for k, v in d.items():
                if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == self.n_agents:
                    per_agent_keys.append(k)

            # If none found, fail with helpful message
            if not per_agent_keys:
                shapes = {k: (list(v.shape) if isinstance(v, np.ndarray) else type(v).__name__) for k, v in d.items()}
                raise AssertionError(
                    f"No stacked per-agent arrays found in dict. Keys={list(d.keys())}, shapes/types={shapes}"
                )

            # Build per-agent vectors from stacked arrays
            per_agent_list: List[List[np.ndarray]] = [[] for _ in range(self.n_agents)]
            for k in per_agent_keys:
                arr = d[k]
                # ensure numeric
                if not np.issubdtype(arr.dtype, np.number):
                    try:
                        arr = arr.astype(np.float32)
                    except Exception:
                        continue
                # slice along first axis
                for i in range(self.n_agents):
                    per_agent_list[i].append(arr[i].reshape(-1).astype(np.float32))

            # Summarize/attach globals (arrays that are NOT stacked)
            global_feats = self._summarize_globals_from_dict(d, exclude_keys=set(per_agent_keys))
            if global_feats.size > 0:
                for i in range(self.n_agents):
                    per_agent_list[i].append(global_feats)

            # Concatenate per agent
            pa = [np.concatenate(chunks, axis=0) if len(chunks) else np.zeros(1, np.float32)
                  for chunks in per_agent_list]
            return pa

        # Unsupported
        raise AssertionError(f"Unsupported obs container type: {type(obs_like)}")

    def _summarize_globals_from_dict(self, d: Dict[str, Any], exclude_keys: set[str]) -> np.ndarray:
        """
        Summarize non-stacked/global entries into a compact vector (same for all agents).
        - For BEV ('bev'/'bird_eye'/'birdseye'):
            * if include_bev: flatten (cap) or stats; else skip.
        - For other arrays: if small -> flatten; if large -> stats (or cap).
        """
        buf: List[np.ndarray] = []
        for k, v in d.items():
            if k in exclude_keys:
                continue
            # Detect BEV-ish keys
            is_bev = k.lower() in ("bev", "bird_eye", "birdseye")
            if isinstance(v, np.ndarray):
                arr = v
                if not np.issubdtype(arr.dtype, np.number):
                    try:
                        arr = arr.astype(np.float32)
                    except Exception:
                        continue
                if is_bev:
                    if not self.include_bev:
                        continue
                    buf.append(self._handle_large_tensor(arr, force_bev=True))
                else:
                    buf.append(self._handle_large_tensor(arr, force_bev=False))
            # ignore non-arrays
        if not buf:
            return np.zeros(0, dtype=np.float32)
        g = np.concatenate([b.reshape(-1).astype(np.float32) for b in buf], axis=0)
        # cap global vector length to max_flatten to keep obs bounded
        if g.size > self.max_flatten:
            g = g[: self.max_flatten]
        return g

    # ------------ Encoding helpers ------------
    def _encode_obs_probe(self, o_any: Any) -> np.ndarray:
        """Flatten numeric content of a per-agent item to 1D; summarize large tensors."""
        # fast path: if already ndarray
        if isinstance(o_any, np.ndarray):
            arr = o_any
            if not np.issubdtype(arr.dtype, np.number):
                try:
                    arr = arr.astype(np.float32)
                except Exception:
                    return np.zeros(1, np.float32)
            flat = arr.reshape(-1).astype(np.float32)
            return self._cap_or_stats(flat, arr)

        # dict/list fallback -> collect numerics
        buf: List[np.ndarray] = []
        self._collect_numeric(o_any, buf, path=())
        if not buf:
            return np.zeros(1, dtype=np.float32)
        x = np.concatenate(buf, axis=0).astype(np.float32)
        return self._cap_or_stats(x, x)

    def _collect_numeric(self, obj: Any, out: List[np.ndarray], path: Tuple[str, ...]):
        if isinstance(obj, (str, bytes)):
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._collect_numeric(v, out, path + (str(k).lower(),))
            return
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                self._collect_numeric(v, out, path + (str(i),))
            return
        arr = obj if isinstance(obj, np.ndarray) else np.asarray(obj)
        if not np.issubdtype(arr.dtype, np.number):
            try:
                arr = arr.astype(np.float32)
            except Exception:
                return
        flat = arr.reshape(-1).astype(np.float32)
        out.append(self._cap_or_stats(flat, arr))

    def _cap_or_stats(self, flat: np.ndarray, original: np.ndarray) -> np.ndarray:
        if flat.size <= self.max_flatten:
            return flat
        # prefer stats for very large tensors
        if self.large_as_stats:
            x = original.astype(np.float32, copy=False)
            if x.ndim == 3 and x.shape[-1] <= 16:
                feats = []
                C = x.shape[-1]
                for c in range(C):
                    xc = x[..., c]
                    feats += [float(xc.mean()), float(xc.std()), float(xc.min()), float(xc.max())]
                return np.asarray(feats, np.float32)
            return np.asarray([x.mean(), x.std(), x.min(), x.max()], np.float32)
        # otherwise cap
        return flat[: self.max_flatten]

    def _handle_large_tensor(self, arr: np.ndarray, force_bev: bool) -> np.ndarray:
        x = arr.astype(np.float32, copy=False)
        flat = x.reshape(-1)
        if self.bev_summary == "flatten" and (force_bev or (flat.size <= self.max_flatten)):
            return flat[: self.max_flatten]
        # stats summary
        if x.ndim == 3 and x.shape[-1] <= 16:
            feats = []
            C = x.shape[-1]
            for c in range(C):
                xc = x[..., c]
                feats += [float(xc.mean()), float(xc.std()), float(xc.min()), float(xc.max())]
            return np.asarray(feats, np.float32)
        return np.asarray([x.mean(), x.std(), x.min(), x.max()], np.float32)

    def _encode_global_raw(self) -> np.ndarray:
        H, W, D = self.env.world.env.map_H, self.env.world.env.map_W, self.env.world.env.map_D
        tfrac = np.array([self.env._t / max(1, self.env.max_steps)], np.float32)
        uavz = np.array([self.env.uav.pos[2] / max(1.0, D - 1)], np.float32)
        if self.env.n_ugv > 0:
            P = np.stack([g.pos[:2] for g in self.env.ugvs], axis=0)
            C = P.mean(axis=0)
            r_group = float(np.max(np.linalg.norm(P - C[None, :], axis=1)))
        else:
            r_group = 0.0
        group = np.array([r_group / max(H, W)], np.float32)
        return np.concatenate([tfrac, uavz, group], axis=0)

    def _encode_global(self) -> np.ndarray:
        g = self._encode_global_raw()
        g = self._fix_size(g, self.gvec_size)
        return self._clip(g)

    def _build_state(self, obs_encoded: List[np.ndarray]) -> np.ndarray:
        g = self._encode_global()
        state = np.concatenate(obs_encoded + [g], axis=0).astype(np.float32)
        state = self._fix_size(state, self.share_obs_size)
        return self._clip(state)

    # ------------ size/clip helpers ------------
    @staticmethod
    def _fix_size(x: np.ndarray, target: int) -> np.ndarray:
        x = x.reshape(-1).astype(np.float32, copy=False)
        n = x.size
        if n == target:
            return x
        if n < target:
            out = np.zeros((target,), dtype=np.float32)
            out[:n] = x
            return out
        return x[:target]

    def _clip(self, x: np.ndarray) -> np.ndarray:
        if self.clip_obs_to_box:
            return np.clip(x, self.low_clip, self.high_clip, dtype=np.float32)
        return x.astype(np.float32, copy=False)
