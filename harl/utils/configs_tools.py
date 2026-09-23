# Tools for loading and updating configs.  (MPE3D-ready)
import time
import os
import json
import yaml
from typing import Any, Dict, Tuple

# -----------------------------
# YAML loading
# -----------------------------
def get_defaults_yaml_args(algo: str, env: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load config file for user-specified algo and env.
    Returns:
        algo_args: Algorithm config dict.
        env_args : Environment config dict.
    """
    base_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    algo_cfg_path = os.path.join(base_path, "configs", "algos_cfgs", f"{algo}.yaml")
    env_cfg_path  = os.path.join(base_path, "configs", "envs_cfgs",  f"{env}.yaml")

    if not os.path.isfile(algo_cfg_path):
        raise FileNotFoundError(f"Algo config not found: {algo_cfg_path}")
    if not os.path.isfile(env_cfg_path):
        raise FileNotFoundError(f"Env config not found:  {env_cfg_path}")

    with open(algo_cfg_path, "r", encoding="utf-8") as f:
        algo_args = yaml.load(f, Loader=yaml.FullLoader) or {}
    with open(env_cfg_path, "r", encoding="utf-8") as f:
        env_args = yaml.load(f, Loader=yaml.FullLoader) or {}
    return algo_args, env_args


# -----------------------------
# CLI overrides -> config dicts
# -----------------------------
def update_args(unparsed_dict: Dict[str, Any], *args):
    """
    Update loaded config dicts with unparsed command-line arguments.
    Only overrides existing keys; nested dicts supported.
    """

    def update_dict(src: Dict[str, Any], dst: Dict[str, Any]):
        for k, v in dst.items():
            if isinstance(v, dict) and isinstance(src.get(k, None), dict):
                update_dict(src[k], dst[k])
            else:
                if k in src:
                    dst[k] = src[k]

    for target in args:
        update_dict(unparsed_dict, target)


# -----------------------------
# Task name resolver
# -----------------------------
def get_task_name(env: str, env_args: Dict[str, Any]) -> str:
    """Return a stable task name used in the results directory layout."""
    if env == "smac":
        task = env_args["map_name"]
    elif env == "smacv2":
        task = env_args["map_name"]
    elif env == "mamujoco":
        task = f"{env_args['scenario']}-{env_args['agent_conf']}"
    elif env == "pettingzoo_mpe":
        task = f"{env_args['scenario']}-{'continuous' if env_args.get('continuous_actions', False) else 'discrete'}"
    elif env == "gym":
        task = env_args["scenario"]
    elif env == "football":
        task = env_args["env_name"]
    elif env == "dexhands":
        task = env_args["task"]
    elif env == "lag":
        task = f"{env_args['scenario']}-{env_args['task']}"
    elif env == "mpe3d_cont":
        # ---- MPE3D continuous (UGV-only) ----
        H, W, D = (env_args.get("map_size") or [100, 100, 30])
        n_ugv = env_args.get("n_ugv", 4)
        n_ev  = env_args.get("n_evader", env_args.get("n_evaders", 2))
        T     = env_args.get("episode_limit", env_args.get("max_steps", 1000))
        task  = f"{H}x{W}x{D}-ugv{n_ugv}-ev{n_ev}-T{T}"
    else:
        # Fallback: stringify known keys to keep path deterministic
        task = str(env_args.get("scenario", env))
    return task


# -----------------------------
# Result dirs & TB writer
# -----------------------------
def init_dir(env: str, env_args: Dict[str, Any], algo: str, exp_name: str, seed: int, logger_path: str):
    """Init directory for saving results."""
    task = get_task_name(env, env_args)
    hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    results_path = os.path.join(
        logger_path, env, task, algo, exp_name, "-".join([f"seed-{seed:0>5}", hms_time])
    )
    log_path = os.path.join(results_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_path)

    models_path = os.path.join(results_path, "models")
    os.makedirs(models_path, exist_ok=True)
    return results_path, log_path, models_path, writer


# -----------------------------
# JSON-safe helpers
# -----------------------------
def is_json_serializable(value: Any) -> bool:
    """Return True if value can be serialized by json.dumps."""
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False


def convert_json(obj: Any):
    """
    Convert obj to a JSON-serializable structure.
    - dict: convert keys/values
    - tuple: convert to list
    - list: convert items
    - functions/classes: stringify by name
    - dataclass/objects: convert __dict__ recursively
    - fallback: str(obj)
    """
    if is_json_serializable(obj):
        return obj

    if isinstance(obj, dict):
        return {convert_json(k): convert_json(v) for k, v in obj.items()}

    if isinstance(obj, tuple):
        return [convert_json(x) for x in obj]  # tuples -> lists

    if isinstance(obj, list):
        return [convert_json(x) for x in obj]

    if hasattr(obj, "__name__") and "lambda" not in getattr(obj, "__name__", ""):
        return convert_json(obj.__name__)

    if hasattr(obj, "__dict__") and obj.__dict__:
        obj_dict = {convert_json(k): convert_json(v) for k, v in obj.__dict__.items()}
        return {str(obj): obj_dict}

    return str(obj)


# -----------------------------
# Save merged configs to disk
# -----------------------------
def save_config(args: Dict[str, Any], algo_args: Dict[str, Any], env_args: Dict[str, Any], run_dir: str):
    """Persist the configuration used for the run into config.json (JSON-friendly)."""
    config = {"main_args": args, "algo_args": algo_args, "env_args": env_args}
    config_json = convert_json(config)
    output = json.dumps(config_json, separators=(",", ":\t"), indent=4, sort_keys=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as out:
        out.write(output)
