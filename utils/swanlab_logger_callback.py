import enum
import os
import pickle
import urllib
from numbers import Number
from pprint import pprint
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow.fs

import ray
from ray import logger
from ray.air.constants import TRAINING_ITERATION
from ray.air.util.node import _force_on_current_node
from ray.train._internal.session import get_session
from ray.train._internal.syncer import DEFAULT_SYNC_TIMEOUT
from ray.tune.experiment import Trial
from ray.tune.logger import LoggerCallback
from ray.tune.utils import flatten_dict
from ray.util.queue import Queue

from config import ConfigSingleton

'''
migrate from the WandbLoggerCallback
https://docs.ray.io/en/latest/tune/api/doc/ray.air.integrations.wandb.WandbLoggerCallback.html
'''

import swanlab

def _is_allowed_type(obj):

    #just allow all type
    return True
    """Return True if type is allowed for logging to swanlab"""
    ##TODO
    # if isinstance(obj, np.ndarray) and obj.size == 1:
    #     return isinstance(obj.item(), Number)
    # if isinstance(obj, Sequence) and len(obj) > 0:
    #     return isinstance(obj[0], (Image, Video, WBValue))
    # return isinstance(obj, (Number, WBValue))


def _clean_log(obj: Any):
    # Fixes https://github.com/ray-project/ray/issues/10631
    if isinstance(obj, dict):
        return {k: _clean_log(v) for k, v in obj.items()}
    elif isinstance(obj, (list, set)):
        return [_clean_log(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_clean_log(v) for v in obj)
    ##TODO
    # elif isinstance(obj, np.ndarray) and obj.ndim == 3:
    #     # Must be single image (H, W, C).
    #     return Image(obj)
    # elif isinstance(obj, np.ndarray) and obj.ndim == 4:
    #     # Must be batch of images (N >= 1, H, W, C).
    #     return (
    #         _clean_log([Image(v) for v in obj]) if obj.shape[0] > 1 else Image(obj[0])
    #     )
    # elif isinstance(obj, np.ndarray) and obj.ndim == 5:
    #     # Must be batch of videos (N >= 1, T, C, W, H).
    #     return (
    #         _clean_log([Video(v) for v in obj]) if obj.shape[0] > 1 else Video(obj[0])
    #     )
    elif _is_allowed_type(obj):
        return obj

    try:
        # This is probably unnecessary, but left here to be extra sure.
        pickle.dumps(obj)

        return obj
    except Exception:
        # give up, similar to _SafeFallBackEncoder
        fallback = str(obj)

        # Try to convert to int
        try:
            fallback = int(fallback)
            return fallback
        except ValueError:
            pass

        # Try to convert to float
        try:
            fallback = float(fallback)
            return fallback
        except ValueError:
            pass

        # Else, return string
        return fallback

'''
_run_swanlab_process_run_info_hook 是一个扩展点，
用于在 W&B 初始化后执行用户自定义的逻辑。它通过环境变量动态加载外部代码，实现了灵活的插件化设计。
swanlab 暂不使用
'''
# def _run_swanlab_process_run_info_hook(run: Any) -> None:
#     """Run external hook to process information about swanlab run"""
#     if WANDB_PROCESS_RUN_INFO_HOOK in os.environ:
#         try:
#             load_class(os.environ[WANDB_PROCESS_RUN_INFO_HOOK])(run)
#         except Exception as e:
#             logger.exception(
#                 f"Error calling {WANDB_PROCESS_RUN_INFO_HOOK}: {e}", exc_info=e
#             )


class _QueueItem(enum.Enum):
    END = enum.auto()
    RESULT = enum.auto()
    CHECKPOINT = enum.auto()


class _SandbLabLoggingActor:
    """
     We use Ray actors as forking multiprocessing
    processes is not supported by Ray and spawn processes run into pickling
    problems.

    We use a queue for the driver to communicate with the logging process.
    The queue accepts the following items:

    - If it's a dict, it is assumed to be a result and will be logged using
      ``swanlab.log()``
    """

    def __init__(
        self,
        logdir: str,
        queue: Queue,
        exclude: List[str],
        to_config: List[str],
        *args,
        **kwargs,
    ):
        import  swanlab
        self._swanlab = swanlab
        os.chdir(logdir)
        self.queue = queue
        self._exclude = set(exclude)
        self._to_config = set(to_config)
        self.args = args
        self.kwargs = kwargs

        self._trial_name = self.kwargs.get("name", "unknown")
        self._logdir = logdir

    def run(self):
        # Since we're running in a separate process already, use threads.
        os.environ["WANDB_START_METHOD"] = "thread"
        run = self._swanlab.init(*self.args, **self.kwargs)
        run.config.trial_log_path = self._logdir
        #log config file
        args = ConfigSingleton().get_args()
        text = self._swanlab.Text(str(args))
        self._swanlab.log(text,print_to_console = True)

        #_run_swanlab_process_run_info_hook(run)
        while True:
            item_type, item_content = self.queue.get()
            if item_type == _QueueItem.END:
                break

            if item_type == _QueueItem.CHECKPOINT:
                self._handle_checkpoint(item_content)
                continue

            assert item_type == _QueueItem.RESULT
            log, config_update = self._handle_result(item_content)
            try:
                self._swanlab.config.update(config_update, allow_val_change=True)
                self._swanlab.log(log, step=log.get(TRAINING_ITERATION))
            except urllib.error.HTTPError as e:
                # Ignore HTTPError. Missing a few data points is not a
                # big issue, as long as things eventually recover.
                logger.warning("Failed to log result to w&b: {}".format(str(e)))
        print('swanlab finish')
        self._swanlab.finish()

    def _handle_checkpoint(self, checkpoint_path: str):
        text = swanlab.Text(checkpoint_path)
        swanlab.log({f"checkpoint": text},step= int(checkpoint_path[-4:]))
        # print(f'logging the checkpoint path, paht={checkpoint_path}')

    def _handle_result(self, result: Dict) -> Tuple[Dict, Dict]:
        config_update = result.get("config", {}).copy()
        log = {}
        flat_result = flatten_dict(result, delimiter="/")
        for k, v in flat_result.items():
            if any(k.startswith(item + "/") or k == item for item in self._exclude):
                continue
            elif any(k.startswith(item + "/") or k == item for item in self._to_config):
                config_update[k] = v
            elif not _is_allowed_type(v):
                continue
            else:
                log[k] = v

        config_update.pop("callbacks", None)  # Remove callbacks
        return log, config_update


class SwanLabLoggerCallback(LoggerCallback):
    """SwanLabLoggerCallback
    Example:
        .. testcode::

            import random

            from ray import tune
            def train_func(config):
                offset = random.random() / 5
                for epoch in range(2, config["epochs"]):
                    acc = 1 - (2 + config["lr"]) ** -epoch - random.random() / epoch - offset
                    loss = (2 + config["lr"]) ** -epoch + random.random() / epoch + offset
                    train.report({"acc": acc, "loss": loss})
            
            
            tune_callbacks=[]
            tune_callbacks.append(SwanLabLoggerCallback(
                api_key=wandb_key,
                project=project,
                workspace='YOUR_WORK_SPACE',
                upload_checkpoints=False,
                **kwargs,
                 )
            )
            tuner = tune.Tuner(
                train_func,
                param_space={
                    "lr": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
                    "epochs": 10,
                }
                run_config=tune.RunConfig(
                    callbacks=tune_callbacks
                ),
            )
            results = tuner.fit()
    Args:
        project: Name of the  project. Mandatory.
        group: Name of the  group. Defaults to the trainable
            name.
        api_key:  API Key. Alternative to setting ``api_key_file``.
        excludes: List of metrics and config that should be excluded from
            the log.
        log_config: Boolean indicating if the ``config`` parameter of
            the ``results`` dict should be logged. This makes sense if
            parameters will change during training, e.g. with
            PopulationBasedTraining. Defaults to False.
            
        **kwargs: The keyword arguments will be pased to ``swanlab.init()``.

     ``group``, ``run_id`` and ``run_name`` are automatically selected
    by Tune, but can be overwritten by filling out the respective configuration
    values.

    """  #

    # Do not log these result keys
    _exclude_results = ["done", "should_checkpoint"]

    AUTO_CONFIG_KEYS = [
        "trial_id",
        "experiment_tag",
        "node_ip",
        "experiment_id",
        "hostname",
        "pid",
        "date",
    ]

    _logger_actor_cls = _SandbLabLoggingActor
    #see https://docs.swanlab.cn/api/py-init.html
    def __init__(
        self,
        project: Optional[str] = None,
        # group: Optional[str] = None,
        workspace: str = 'default',
        api_key_file: Optional[str] = None,
        api_key: Optional[str] = None,
        excludes: Optional[List[str]] = None,
        log_config: bool = False,
        upload_checkpoints: bool = False,
        upload_timeout: int = DEFAULT_SYNC_TIMEOUT,
        **kwargs,
    ):
        if not swanlab:
            raise RuntimeError(
                "swanlab was not found - please install with `pip install swanlab`"
            )

        self.project = project
        self.workspace = workspace
        self.api_key_path = api_key_file
        self.api_key = api_key
        self.excludes = excludes or []
        self.log_config = log_config
        self.upload_checkpoints = upload_checkpoints
        self._upload_timeout = upload_timeout
        self.kwargs = kwargs

        self._remote_logger_class = None

        self._trial_logging_actors: Dict[
            "Trial", ray.actor.ActorHandle[_SandbLabLoggingActor]
        ] = {}
        self._trial_logging_futures: Dict["Trial", ray.ObjectRef] = {}
        self._logging_future_to_trial: Dict[ray.ObjectRef, "Trial"] = {}
        self._trial_queues: Dict["Trial", Queue] = {}


    def log_trial_start(self, trial: "Trial"):
        config = trial.config.copy()

        config.pop("callbacks", None)  # Remove callbacks

        exclude_results = self._exclude_results.copy()

        # Additional excludes
        exclude_results += self.excludes

        # Log config keys on each result?
        if not self.log_config:
            exclude_results += ["config"]

        # Fill trial ID and name
        trial_id = trial.trial_id if trial else None
        trial_name = str(trial) if trial else None

        # Project name for Wandb
        swanlab_project = self.project

        swanlab_workspace = self.workspace

        # remove unpickleable items!
        config = _clean_log(config)
        config = {
            key: value for key, value in config.items() if key not in self.excludes
        }

        swanlab_init_kwargs = dict(
            id=trial_id,
            name=trial_name,
            resume=False,
            reinit=True,
            allow_val_change=True,
            workspace=swanlab_workspace,
            project=swanlab_project,
            config=config,
        )
        swanlab_init_kwargs.update(self.kwargs)

        self._start_logging_actor(trial, exclude_results, **swanlab_init_kwargs)

    def _start_logging_actor(
        self, trial: "Trial", exclude_results: List[str], **swanlab_init_kwargs
    ):
        try:
            # Reuse actor if one already exists.
            # This can happen if the trial is restarted.
            if trial in self._trial_logging_futures:
                return

            if not self._remote_logger_class:
                env_vars = {}
                #TODO  check api key
                self._remote_logger_class = ray.remote(
                    num_cpus=0,
                    **_force_on_current_node(),
                    runtime_env={"env_vars": env_vars},
                    max_restarts=-1,
                    max_task_retries=-1,
                )(self._logger_actor_cls)

            self._trial_queues[trial] = Queue(
                actor_options={
                    "num_cpus": 0,
                    **_force_on_current_node(),
                    "max_restarts": -1,
                    "max_task_retries": -1,
                }
            )

            self._trial_logging_actors[trial] = self._remote_logger_class.remote(
                logdir=trial.local_path,
                queue=self._trial_queues[trial],
                exclude=exclude_results,
                to_config=self.AUTO_CONFIG_KEYS,
                **swanlab_init_kwargs,
            )

            logging_future = self._trial_logging_actors[trial].run.remote()
            self._trial_logging_futures[trial] = logging_future
            self._logging_future_to_trial[logging_future] = trial
        except Exception as e:
            print(f'fail to  self._remote_logger_class.remote:{e}')

    def _signal_logging_actor_stop(self, trial: "Trial"):
        self._trial_queues[trial].put((_QueueItem.END, None))

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        if trial not in self._trial_logging_actors:
            self.log_trial_start(trial)

        result = _clean_log(result)
        self._trial_queues[trial].put((_QueueItem.RESULT, result))

    def log_trial_save(self, trial: "Trial"):
        if self.upload_checkpoints and trial.checkpoint:
            checkpoint_root = None
            if isinstance(trial.checkpoint.filesystem, pyarrow.fs.LocalFileSystem):
                checkpoint_root = trial.checkpoint.path

            if checkpoint_root:
                self._trial_queues[trial].put((_QueueItem.CHECKPOINT, checkpoint_root))

    def log_trial_end(self, trial: "Trial", failed: bool = False):

        self._signal_logging_actor_stop(trial=trial)
        self._cleanup_logging_actors()

    def _cleanup_logging_actor(self, trial: "Trial"):
        del self._trial_queues[trial]
        del self._trial_logging_futures[trial]
        ray.kill(self._trial_logging_actors[trial])
        del self._trial_logging_actors[trial]

    def _cleanup_logging_actors(self, timeout: int = 0, kill_on_timeout: bool = False):
        """Clean up logging actors that have finished.
        Waits for `timeout` seconds to collect finished logging actors.

        Args:
            timeout: The number of seconds to wait. Defaults to 0 to clean up
                any immediate logging actors during the run.
                This is set to a timeout threshold to wait for pending uploads
                on experiment end.
            kill_on_timeout: Whether or not to kill and cleanup the logging actor if
                it hasn't finished within the timeout.
        """

        futures = list(self._trial_logging_futures.values())
        done, remaining = ray.wait(futures, num_returns=len(futures), timeout=timeout)
        for ready_future in done:
            finished_trial = self._logging_future_to_trial.pop(ready_future)
            self._cleanup_logging_actor(finished_trial)

        if kill_on_timeout:
            for remaining_future in remaining:
                trial = self._logging_future_to_trial.pop(remaining_future)
                self._cleanup_logging_actor(trial)

    def on_experiment_end(self, trials: List["Trial"], **info):
        """Wait for the actors to finish their call to `swanlab.finish`.
        This includes uploading all logs + artifacts to swanlab."""
        self._cleanup_logging_actors(timeout=self._upload_timeout, kill_on_timeout=True)

    def __del__(self):
        if ray.is_initialized():
            for trial in list(self._trial_logging_actors):
                self._signal_logging_actor_stop(trial=trial)

            self._cleanup_logging_actors(timeout=2, kill_on_timeout=True)

        self._trial_logging_actors = {}
        self._trial_logging_futures = {}
        self._logging_future_to_trial = {}
        self._trial_queues = {}
