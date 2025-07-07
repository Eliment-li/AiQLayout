import re
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
    smd = SharedMemoryDict(name='ConfigSingleton', size=10240)

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




    # def to_string(self):
    #     """
    #     返回配置的字符串表示
    #     """
    #     return OmegaConf.to_yaml(self.args)
    # def update(self, key, value):
    #     """
    #     动态添加键值对到配置中
    #     """
    #     keys = key.split(".")
    #     current = self.args
    #     for k in keys[:-1]:
    #         if k not in current:
    #             current[k] = {}  # 创建嵌套字典
    #         current = current[k]
    #     current[keys[-1]] = value
