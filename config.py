from hydra import initialize, compose
from threading import Lock

from omegaconf import OmegaConf


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

    def custom_configure(self):
        pass


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

logical_cores = get_logical_cores()
print(f"逻辑核心数量: {logical_cores}")


# 示例使用
if __name__ == "__main__":
    # 第一次初始化
    global_config = ConfigSingleton()
    config = global_config.get_args()
    print(config)

    # 再次获取，确保是同一个实例
    another_config_instance = ConfigSingleton()
    print(another_config_instance.get_args())

    # 验证单例
    print(global_config is another_config_instance)  # 输出: True
