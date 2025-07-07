import os
import re
from datetime import datetime
from pathlib import Path

import ray
import yaml
import redis
import threading
from glob import glob

from utils.file.file_util import get_root_dir
import torch


def enhance_base_config(config,args):
    # Disable the new API stack
    # @see https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.api_stack.html#ray-rllib-algorithms-algorithm-config-algorithmconfig-api-stack

    if args.num_env_runners is not None:
        config.env_runners(num_env_runners=args.num_env_runners)
    if args.num_envs_per_env_runner is not None:
        config.env_runners(num_envs_per_env_runner=args.num_envs_per_env_runner)

    args.num_learners = 0
    # New stack.
    if args.enable_new_api_stack:
        # GPUs available in the cluster?
        num_gpus_available = ray.cluster_resources().get("GPU", 0)
        print("num_gpus_available: ", num_gpus_available)
        num_gpus_requested = args.num_gpus_per_learner * args.num_learners

        # Define compute resources used.
        #config.resources(num_gpus=num_gpus_available)  # old API stack setting

        #set num_learners and num_gpus_per_learner
        config.learners(num_learners=args.num_learners)
        if num_gpus_available >= num_gpus_requested:
            # All required GPUs are available -> Use them.
            config.learners(num_gpus_per_learner=args.num_gpus_per_learner)
        else:
            config.learners(num_gpus_per_learner=0)
            print(
                "Warning! You are running your script with --num-learners="
                f"{args.num_learners} and --num-gpus-per-learner="
                f"{args.num_gpus_per_learner}, but your cluster only has "
                f"{num_gpus_available} GPUs!"
            )
        # Set CPUs per Learner.
        if args.num_cpus_per_learner is not None:
            config.learners(num_cpus_per_learner=args.num_cpus_per_learner)

    else:
        config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )

    # Evaluation setup.
    if args.evaluation_interval > 0:
        config.evaluation(
            evaluation_num_env_runners=args.evaluation_num_env_runners,
            evaluation_interval=args.evaluation_interval,
            evaluation_duration=args.evaluation_duration,
            evaluation_duration_unit=args.evaluation_duration_unit,
            evaluation_parallel_to_training=args.evaluation_parallel_to_training,
        )

    # Set the log-level (if applicable).
    if args.log_level is not None:
        config.debugging(log_level=args.log_level)

    # Set the output dir (if applicable).
    if args.output is not None:
        config.offline_data(output=args.output)


def dynamic_conf(exist_conf):
    # 0.8 * logical_cpu_cores
    # cfg['num_cpus'] = int(os.cpu_count() * 0.7)
    cfg={}
    # default number_learner is 1, thus  set num_gpus_per_learner to 1 is fine
    cfg['num_gpus_per_learner'] = 1 if torch.cuda.is_available() else 0
    cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg['time_id'] = datetime.now().strftime('%m%d_%H%M')
    cfg['output'] = None
    #cfg['tensorboard_path'] = p / 'results' / 'tensorboard'
    path = Path(get_root_dir()) / 'results' / 'evaluate' / cfg['time_id']
    cfg['results_evaluate_path'] = str(path)
    os.makedirs(path, exist_ok=True)
    match = re.search(r"LSI.*", exist_conf['lsi_file_path'])
    circuit = match.group()
    cfg['wandb_run_name'] = cfg['time_id'] + '_' + circuit
    return cfg

#update dynamic config before start a new experiment
def update_for_new_exp():
    conf = get_args()

    time_id = datetime.now().strftime('%m%d_%H%M')
    path = Path(get_root_dir()) / 'results' / 'evaluate' / time_id
    results_evaluate_path = str(path)
    os.makedirs(path, exist_ok=True)
    match = re.search(r"LSI.*", conf.lsi_file_path)
    circuit = match.group()

    conf.update('time_id', time_id)
    conf.update('results_evaluate_path', results_evaluate_path)
    conf.update('wandb_run_name', time_id + '_' + circuit)


def merge_dict(a, b):
    """
    递归合并b到a, b的值覆盖a的值
    """
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            merge_dict(a[k], v)
        else:
            a[k] = v

class ConfigNode:
    def __init__(self, data):
        for k, v in data.items():
            if isinstance(v, dict):
                v = ConfigNode(v)
            setattr(self, k, v)
        self._data = data

    def __getitem__(self, key):
        return getattr(self, key)

    def __setattr__(self, key, value):
        if key != '_data' and hasattr(self, '_data'):
            self._data[key] = value
        object.__setattr__(self, key, value)

    def to_dict(self):
        result = {}
        for k, v in self._data.items():
            if isinstance(v, ConfigNode):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

class GlobalConfig:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, redis_url='redis://localhost:6379/0', config_dir=None, redis_key='global_config'):
        self.redis_url = redis_url
        self.redis_key = redis_key
        self.config_dir = config_dir
        self._redis = redis.Redis.from_url(self.redis_url, decode_responses=True)
        self._redis.delete(self.redis_key)
        self._load_config()

    def _load_config(self):
        # 如果redis中无配置，则从本地所有yaml加载并写入redis
        if not self._redis.exists(self.redis_key):
            config = {}
            files = sorted(glob(os.path.join(self.config_dir, '*.yml')) + glob(os.path.join(self.config_dir, '*.yaml')))
            if not files:
                raise FileNotFoundError("No yaml config files found in directory")
            for file in files:
                print(f"Loading config from {file}")
                with open(file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                merge_dict(config, data)

            merge_dict(config,dynamic_conf(config))
            #enhance config
            
            self._redis.set(self.redis_key, yaml.dump(config))
        # 从redis获取配置
        config_str = self._redis.get(self.redis_key)
        self._data = yaml.safe_load(config_str)
        self._node = ConfigNode(self._data)

    def __getattr__(self, key):
        return getattr(self._node, key)

    def update(self, key, value):
        # 支持 'a.b.c' 形式的key
        keys = key.split('.')
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        # 更新redis
        self._redis.set(self.redis_key, yaml.dump(self._data))
        # 更新本地node
        self._node = ConfigNode(self._data)

    def reload(self):
        """ 从redis重新加载配置 """
        self._load_config()

    def to_dict(self):
        return self._node.to_dict()

# 全局单例
_config_instance = None

def get_args():
    config_path = Path(get_root_dir(), 'conf')
    print(config_path)
    global _config_instance
    if _config_instance is None:
        _config_instance = GlobalConfig(redis_url='redis://localhost:6379/0', config_dir=config_path, redis_key='global_config')
    else:
        _config_instance.reload()
    return _config_instance

if __name__ == '__main__':
    #from global_config import get_args

    conf = get_args()
    print(conf.lsi_file_path)
    conf.update('wandb_run_name','temp')
    print(conf.wandb_run_name)
    #print(conf.to_dict())