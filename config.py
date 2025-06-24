from datetime import datetime
from pathlib import Path

import ray
import torch
from hydra import initialize, compose
from threading import Lock
from omegaconf import OmegaConf

from utils.file.file_util import get_root_dir
from shared_memory_dict import SharedMemoryDict

class ConfigSingleton:
    _instance = None
    _lock = Lock()
    smd = SharedMemoryDict(name='ConfigSingleton', size=1024)

    def __new__(cls, config_path="conf", job_name="config",version_base="1.2"):
        # 使用线程锁确保线程安全
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize(config_path,job_name,version_base)
            return cls._instance

    def _initialize(self, config_path, job_name, version_base):
        # 初始化 hydra 配置
        with initialize(config_path=config_path,job_name=job_name,version_base=version_base):
            # 如果有 overrides，可以在这里传递覆盖参数
            self.config = compose(config_name=job_name, overrides= [])
            self.args = OmegaConf.create({})
            for key in self.config.keys():
                self.args = OmegaConf.merge(self.args, self.config[key])
            for key, value in self.args.items():
                if str(value).lower() == 'none':
                    # 如果值是字符串 'none'（忽略大小写），则替换为 None
                    self.args[key] = None
            # 处理其他配置项
        self.enhance_configure()

    def enhance_configure(self):
        p = Path(get_root_dir())
        #0.8 * logical_cpu_cores
        # self.args['num_cpus'] = int(os.cpu_count() * 0.7)

        # default number_learner is 1, thus  set num_gpus_per_learner to 1 is fine
        self.args['num_gpus_per_learner'] = 1 if torch.cuda.is_available() else 0
        self.args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        self.args['time_id'] = datetime.now().strftime('%m%d_%H%M')
        self.args['output'] = None
        self.args['tensorboard_path'] = p / 'results' / 'tensorboard'

        if 'results_evaluate_path' in self.smd:
            self.args['results_evaluate_path'] = self.smd['results_evaluate_path']
        else:
            path = Path(get_root_dir()) / 'results' / 'evaluate' / self.args.time_id
            self.args['results_evaluate_path'] = path
            self.smd['results_evaluate_path'] = path
            os.makedirs(path, exist_ok=True)

        if 'wandb_run_name' in self.smd:
            self.args['wandb_run_name'] = self.smd['wandb_run_name']
        else:
            self.args['wandb_run_name'] = self.args.time_id
            self.smd['wandb_run_name'] = self.args.time_id


    def get_args(self):
        return self.args

    def to_string(self):
        """
        返回配置的字符串表示
        """
        return OmegaConf.to_yaml(self.args)
    @staticmethod
    def update(self, key, value):
        """
        动态添加键值对到配置中
        """
        keys = key.split(".")
        current = self.args
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}  # 创建嵌套字典
            current = current[k]
        current[keys[-1]] = value


import os
import multiprocessing

def get_logical_cores():
    try:
        # 优先使用 multiprocessing.cpu_count()
        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        try:
            # 回退到 os.cpu_count()
            return os.cpu_count()
        except (AttributeError, NotImplementedError):
            # 如果都无法获取，返回默认值（通常为1）
            return 1


# config:
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


# test code
if __name__ == "__main__":
    # 第一次初始化
    global_config = ConfigSingleton()
    args = global_config.get_args()
    print(global_config.to_string())


