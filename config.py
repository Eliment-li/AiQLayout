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

#
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


def merge_dict(a, b):
    """
    递归合并b到a, b的值覆盖a的值
    """
    for k, v in b.items():
        if k in a and isinstance(a[k], dict) and isinstance(v, dict):
            merge_dict(a[k], v)
        else:
            a[k] = v

import os
import time
import yaml
import redis
from glob import glob
from collections.abc import MutableMapping

class RedisConfig(MutableMapping):
    def __init__(self, redis_url='redis://localhost:6379/0', redis_key='app_config', init_flag_key='app_config_init'):
        self.redis = redis.StrictRedis.from_url(redis_url, decode_responses=True)
        self.redis_key = redis_key
        self.init_flag_key = init_flag_key
        self._cache = None

    def flush(self):
        print('Flushing Redis database...')
        # clean db 0
        self.redis.flushdb()
    def load_and_merge_yaml(self, config_dir):
        merged = {}
        for file_path in glob(os.path.join(config_dir, '*.yml')) + glob(os.path.join(config_dir, '*.yaml')):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                self.deep_update(merged, data)
        return merged

    @staticmethod
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = RedisConfig.deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def initialize(self, config_dir):
        # 1. 加载并合并配置
        config = self.load_and_merge_yaml(config_dir)
        # 2. 写入 redis
        self.redis.set(self.redis_key, yaml.dump(config))
        # 3. 写入初始化标志
        self.redis.set(self.init_flag_key, '1')
        print('Configuration initialized and written to Redis.')

    def wait_until_initialized(self, timeout=10):
        start = time.time()
        while not self.redis.get(self.init_flag_key):
            if time.time() - start > timeout:
                raise TimeoutError('Initialization flag not set in Redis')
            time.sleep(0.5)
        self._load_from_redis()

    def _load_from_redis(self):
        config_str = self.redis.get(self.redis_key)
        if not config_str:
            raise ValueError('No config found in Redis')
        self._cache = yaml.safe_load(config_str)

    def __getitem__(self, key):
        if self._cache is None:
            self._load_from_redis()
        return self._cache[key]

    def __setitem__(self, key, value):
        if self._cache is None:
            self._load_from_redis()
        self._cache[key] = value
        self.redis.set(self.redis_key, yaml.dump(self._cache))

    def __delitem__(self, key):
        if self._cache is None:
            self._load_from_redis()
        del self._cache[key]
        self.redis.set(self.redis_key, yaml.dump(self._cache))

    def __iter__(self):
        if self._cache is None:
            self._load_from_redis()
        return iter(self._cache)

    def __len__(self):
        if self._cache is None:
            self._load_from_redis()
        return len(self._cache)

    def __getattr__(self, name):
        if self._cache is None:
            self._load_from_redis()
        try:
            return self._cache[name]
        except KeyError:
            raise AttributeError(f"No such key: {name}")

    def update_redis(self, key, value=None):
        for k, v in key.items():
            self[k] = v


    def clear_redis(self):
        self.redis.delete(self.redis_key)
        self.redis.delete(self.init_flag_key)
        self._cache = None


if __name__ == '__main__':
    path = Path(get_root_dir()) / 'conf'
    redis_config = RedisConfig()
    redis_config.flush()  # 清空 Redis 数据库
    # 初始化（只做一次）
    redis_config.initialize(path)

    # 其他进程/线程中加载并等待初始化
    rc = RedisConfig()
    rc.wait_until_initialized()
    print(rc['gamma'])
    print(rc.gamma)

    # 更新
    rc.update_redis('gamma', 'new_value')
    print(rc.gamma)
    # 清理
    rc.clear_redis()