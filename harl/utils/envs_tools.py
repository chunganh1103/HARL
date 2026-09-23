"""Tools for HARL."""
import os
import random
import numpy as np
import torch
from harl.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def check(value):
    """Convert numpy to torch tensor if needed."""
    return torch.from_numpy(value) if isinstance(value, np.ndarray) else value


def get_shape_from_obs_space(obs_space):
    """Return shape from observation space (Box or list)."""
    if obs_space.__class__.__name__ == "Box":
        return obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        return obs_space
    raise NotImplementedError


def get_shape_from_act_space(act_space):
    """Return action dimension from action space."""
    if act_space.__class__.__name__ == "Discrete":
        return 1
    if act_space.__class__.__name__ == "MultiDiscrete":
        return act_space.shape[0]
    if act_space.__class__.__name__ == "Box":
        return act_space.shape[0]
    if act_space.__class__.__name__ == "MultiBinary":
        return act_space.shape[0]
    raise NotImplementedError


def make_train_env(env_name, seed, n_threads, env_args):
    """Make env for training."""
    if env_name == "dexhands":
        from harl.envs.dexhands.dexhands_env import DexHandsEnv
        return DexHandsEnv({"n_threads": n_threads, **env_args})

    # ---- mpe3d_cont: vectorized like other envs ----
    if env_name == "mpe3d_cont":
        from harl.envs.mpe3d.mpe3d_cont import MPE3DContHARL

        def get_env_fn(rank):
            def init_env():
                env = MPE3DContHARL(env_args)
                # wrapper has .seed(); also keep reproducible per-rank offset
                if hasattr(env, "seed"):
                    env.seed(seed + rank * 1000)
                return env
            return init_env

        if n_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])

    # ---- default vectorized path for other envs ----
    def get_env_fn(rank):
        def init_env():
            if env_name == "smac":
                from harl.envs.smac.StarCraft2_Env import StarCraft2Env
                env = StarCraft2Env(env_args)
            elif env_name == "smacv2":
                from harl.envs.smacv2.smacv2_env import SMACv2Env
                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
                env = MujocoMulti(env_args=env_args)
            elif env_name == "pettingzoo_mpe":
                from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
                assert env_args["scenario"] in [
                    "simple_v2",
                    "simple_spread_v2",
                    "simple_reference_v2",
                    "simple_speaker_listener_v3",
                ], "only cooperative scenarios in MPE are supported"
                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from harl.envs.gym.gym_env import GYMEnv
                env = GYMEnv(env_args)
            elif env_name == "football":
                from harl.envs.football.football_env import FootballEnv
                env = FootballEnv(env_args)
            elif env_name == "lag":
                from harl.envs.lag.lag_env import LAGEnv
                env = LAGEnv(env_args)
            else:
                print("Can not support the " + env_name + " environment.")
                raise NotImplementedError
            env.seed(seed + rank * 1000)
            return env
        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""
    if env_name == "dexhands":
        raise NotImplementedError  # dexhands does not support multi-instance eval

    # ---- mpe3d_cont: vectorized like other envs ----
    if env_name == "mpe3d_cont":
        from harl.envs.mpe3d.mpe3d_cont import MPE3DContHARL

        def get_env_fn(rank):
            def init_env():
                env = MPE3DContHARL(env_args)
                if hasattr(env, "seed"):
                    env.seed(seed * 50000 + rank * 10000)
                return env
            return init_env

        if n_threads == 1:
            return ShareDummyVecEnv([get_env_fn(0)])
        else:
            return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])

    def get_env_fn(rank):
        def init_env():
            if env_name == "smac":
                from harl.envs.smac.StarCraft2_Env import StarCraft2Env
                env = StarCraft2Env(env_args)
            elif env_name == "smacv2":
                from harl.envs.smacv2.smacv2_env import SMACv2Env
                env = SMACv2Env(env_args)
            elif env_name == "mamujoco":
                from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
                env = MujocoMulti(env_args=env_args)
            elif env_name == "pettingzoo_mpe":
                from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
                env = PettingZooMPEEnv(env_args)
            elif env_name == "gym":
                from harl.envs.gym.gym_env import GYMEnv
                env = GYMEnv(env_args)
            elif env_name == "football":
                from harl.envs.football.football_env import FootballEnv
                env = FootballEnv(env_args)
            elif env_name == "lag":
                from harl.envs.lag.lag_env import LAGEnv
                env = LAGEnv(env_args)
            else:
                print("Can not support the " + env_name + " environment.")
                raise NotImplementedError
            env.seed(seed * 50000 + rank * 10000)
            return env
        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    """Make env for rendering."""
    manual_render = True
    manual_expand_dims = True
    manual_delay = True
    env_num = 1
    if env_name == "smac":
        from harl.envs.smac.StarCraft2_Env import StarCraft2Env
        env = StarCraft2Env(args=env_args)
        manual_render = False
        manual_delay = False
        env.seed(seed * 60000)
    elif env_name == "smacv2":
        from harl.envs.smacv2.smacv2_env import SMACv2Env
        env = SMACv2Env(args=env_args)
        manual_render = False
        manual_delay = False
        env.seed(seed * 60000)
    elif env_name == "mamujoco":
        from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
        env = MujocoMulti(env_args=env_args)
        env.seed(seed * 60000)
    elif env_name == "pettingzoo_mpe":
        from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv
        env = PettingZooMPEEnv({**env_args, "render_mode": "human"})
        env.seed(seed * 60000)
    elif env_name == "gym":
        from harl.envs.gym.gym_env import GYMEnv
        env = GYMEnv(env_args)
        env.seed(seed * 60000)
    elif env_name == "football":
        from harl.envs.football.football_env import FootballEnv
        env = FootballEnv(env_args)
        manual_render = False
        env.seed(seed * 60000)
    elif env_name == "dexhands":
        from harl.envs.dexhands.dexhands_env import DexHandsEnv
        env = DexHandsEnv({"n_threads": 64, **env_args})
        manual_render = False
        manual_expand_dims = False
        manual_delay = False
        env_num = 64
    elif env_name == "lag":
        from harl.envs.lag.lag_env import LAGEnv
        env = LAGEnv(env_args)
        env.seed(seed * 60000)
    elif env_name == "mpe3d_cont":
        from harl.envs.mpe3d.mpe3d_cont import MPE3DContHARL
        env = MPE3DContHARL({**env_args, "record_video_2d": True, "record_video_3d": True})
        if hasattr(env, "seed"):
            env.seed(seed * 60000)
    else:
        print("Can not support the " + env_name + " environment.")
        raise NotImplementedError
    return env, manual_render, manual_expand_dims, manual_delay, env_num


def set_seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def get_num_agents(env, env_args, envs):
    """Get the number of agents in the environment."""
    if env == "smac":
        from harl.envs.smac.smac_maps import get_map_params
        return get_map_params(env_args["map_name"])["n_agents"]
    elif env == "smacv2":
        return envs.n_agents
    elif env == "mamujoco":
        return envs.n_agents
    elif env == "pettingzoo_mpe":
        return envs.n_agents
    elif env == "gym":
        return envs.n_agents
    elif env == "football":
        return envs.n_agents
    elif env == "dexhands":
        return envs.n_agents
    elif env == "lag":
        return envs.n_agents
    elif env == "mpe3d_cont":
        return envs.n_agents
    else:
        raise NotImplementedError(f"Unknown env for get_num_agents: {env}")
