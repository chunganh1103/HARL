# file: examples/eval_mpe3d_multi.py
import os, sys, glob, time, argparse
from typing import List, Optional, Sequence
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harl.utils.envs_tools import make_render_env


# --------- IO helpers ---------
def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32, copy=False).reshape(1, -1))

def _load_actor_file(path: str):
    try:
        m = torch.jit.load(path, map_location="cpu"); m.eval(); return m
    except Exception:
        pass
    try:
        m = torch.load(path, map_location="cpu")
        if hasattr(m, "eval"): m.eval(); return m
    except Exception:
        pass
    return None

def _pick_agent_files(models_dir: str, n_agents: int, explicit: Optional[Sequence[str]], pattern: Optional[str]) -> List[str]:
    if explicit and len(explicit) > 0:
        files = [f for f in explicit if os.path.isfile(f)]
        if len(files) != len(explicit): print("[warn] some --actor_files not found; ignoring missing.")
        return files

    pats = [
        pattern or "",  # user pattern has priority if provided (e.g. "actor_agent{idx}.pt")
        "actor_agent{idx}.*",
        "agent{idx}_actor.*",
        "actor_{idx}.*",
        "policy{idx}.*",
        "agent{idx}.*actor.*",
    ]
    chosen = []
    for idx in range(n_agents):
        hit = None
        for pat in pats:
            if not pat: continue
            g = os.path.join(models_dir, pat.format(idx=idx))
            cand = glob.glob(g)
            cand = [c for c in cand if os.path.isfile(c)]
            if cand:
                # pick latest by mtime
                cand.sort(key=lambda f: os.path.getmtime(f), reverse=True)
                hit = cand[0]; break
        if hit is None:
            # generic fallback: any actor*.pt files, newest N
            generic = []
            for p in ["actor*.jit", "actor*.pt", "actor*.pth", "policy*.pt", "policy*.jit"]:
                generic += glob.glob(os.path.join(models_dir, p))
            generic = sorted(set([g for g in generic if os.path.isfile(g)]), key=lambda f: os.path.getmtime(f), reverse=True)
            if len(generic) >= n_agents:
                chosen = generic[:n_agents]
                return chosen
            else:
                print(f"[warn] cannot find per-agent file for idx={idx}; will use zeros for this agent.")
                chosen.append("")  # placeholder -> zero action
        else:
            chosen.append(hit)
    return chosen

def _load_actors(models_dir: str, n_agents: int, actor_files: Optional[Sequence[str]], pattern: Optional[str]):
    paths = _pick_agent_files(models_dir, n_agents, actor_files, pattern)
    actors = []
    for i, p in enumerate(paths):
        if not p:
            actors.append(None); continue
        m = _load_actor_file(p)
        if m is None:
            print(f"[warn] failed to load actor for agent {i}: {p}")
        else:
            print(f"[eval] agent {i} loaded: {p}")
        actors.append(m)
    return actors

def _act_single(m, obs_i: np.ndarray) -> np.ndarray:
    if m is None: return np.zeros((2,), dtype=np.float32)
    with torch.no_grad():
        x = _to_tensor(obs_i)
        if hasattr(m, "act"):
            out = m.act(x, deterministic=True)
            if isinstance(out, tuple): out = out[0]
        else:
            out = m(x)
        a = out.detach().cpu().numpy().reshape(-1)
        if a.size < 2: 
            # pad if some model outputs scalar; avoid crash
            a = np.pad(a, (0, 2 - a.size))
        a = a[:2]
        return np.clip(a.astype(np.float32), -1.0, 1.0)

# --------- CLI ---------
def parse_args():
    p = argparse.ArgumentParser("Evaluate per-agent actors on MPE3D and save 2D/3D videos")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--models_dir", type=str, required=True, help="directory containing per-agent actors")
    p.add_argument("--actor_files", type=str, nargs="*", default=None, help="explicit list of actor files in agent order")
    p.add_argument("--actor_pattern", type=str, default=None, help="glob with {idx}, e.g., 'actor_agent{idx}.pt'")
    p.add_argument("--video_dir", type=str, default="./videos_eval")
    # env knobs
    p.add_argument("--n_ugv", type=int, default=4)
    p.add_argument("--n_evader", type=int, default=2)
    p.add_argument("--map_size", type=int, nargs=3, default=[100, 100, 30])
    p.add_argument("--include_bev", action="store_true", default=False)
    p.add_argument("--bev_summary", type=str, default="stats", choices=["stats","flatten"])
    return p.parse_args()

# --------- main ---------
def main():
    args = parse_args()
    os.makedirs(args.video_dir, exist_ok=True)

    env_args = {
        "n_ugv": args.n_ugv,
        "n_evader": args.n_evader,
        "map_size": args.map_size,
        "episode_limit": args.max_steps,
        "record_video_2d": True,
        "record_video_3d": True,
        "video_file": os.path.join(args.video_dir, "mpe3d_eval_3d.mp4"),
        "video2d_file": os.path.join(args.video_dir, "mpe3d_eval_2d.mp4"),
        "include_bev": args.include_bev,
        "bev_summary": args.bev_summary,
    }
    env, manual_render, manual_expand_dims, manual_delay, env_num = make_render_env(
        env_name="mpe3d_cont", seed=args.seed, env_args=env_args
    )
    n = int(getattr(env, "n_agents", args.n_ugv))
    print(f"[eval] n_agents={n}  obs_dim={env.observation_space[0].shape[0]}")

    actors = _load_actors(args.models_dir, n, args.actor_files, args.actor_pattern)

    try:
        for ep in range(args.episodes):
            obs, share, _ = env.reset()
            assert len(obs) == n, f"env returned {len(obs)} obs, expected {n}"
            ep_ret, ep_len = 0.0, 0
            t0 = time.time()
            while ep_len < args.max_steps:
                acts = np.stack([_act_single(actors[i], obs[i]) for i in range(n)], axis=0)  # (n,2)
                obs, share, rews, dones, infos, _ = env.step(acts)
                ep_ret += float(np.mean([float(np.asarray(r)) for r in rews]))
                ep_len += 1
                if manual_render: env.render()
                if all(dones): break
            dt = time.time() - t0
            print(f"[eval] episode {ep+1}/{args.episodes} len={ep_len} ret≈{ep_ret:.2f} time={dt:.2f}s")
    finally:
        env.close()
        print(f"[eval] saved 3D: {os.path.join(args.video_dir, 'mpe3d_eval_3d.mp4')}")
        print(f"[eval] saved 2D: {os.path.join(args.video_dir, 'mpe3d_eval_2d.mp4')}")

if __name__ == "__main__":
    main()
