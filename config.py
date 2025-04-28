from datetime import datetime
from pathlib import Path
import torch
from hydra import initialize, compose
from threading import Lock
from omegaconf import OmegaConf
from utils.file_util import get_root_dir

class ConfigSingleton:
    _instance = None
    _lock = Lock()


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
        # default number_learner is 1, thus  set num_gpus_per_learner to 1 is fine
        self.args['num_gpus_per_learner'] = 1 if torch.cuda.is_available() else 0
        self.args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        self.args['time_id'] = datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.args['output'] = None
        self.args['tensorboard_path'] = p / 'results' / 'tensorboard'


    def get_args(self):
        return self.args
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


10-4-5003

# test code
if __name__ == "__main__":
    # 第一次初始化
    global_config = ConfigSingleton()
    args = global_config.get_args()
    args.num_gpus_per_learner = 2
    print(args.num_gpus_per_learner)
    # get again
    another_config_instance = ConfigSingleton()
    print(another_config_instance.get_args())

    # verify Singleton
    print(global_config is another_config_instance)  # True


