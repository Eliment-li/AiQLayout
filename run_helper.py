import argparse
import logging
import os
import re
import time
from typing import (
    Dict,
    List,
    Optional,
    Union,
)
from gymnasium import register
import numpy as np
import ray
from ray import tune
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.typing import ResultDict
from ray.tune import CLIReporter
from ray.tune.result import TRAINING_ITERATION

from config import enhance_base_config
from ray.rllib.algorithms import AlgorithmConfig
from utils.checkpoint import CheckPointCallback
from utils.swanlab.swanlab_logger_callback import SwanLabLoggerCallback

jax, _ = try_import_jax()
tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

logger = logging.getLogger(__name__)
def trial_str_creator(trial):
    return "{}_{}".format(trial.trainable_name, trial.trial_id)

def train_no_tune(args, config, stop: Optional[Dict] = None):
    results = None
    assert not args.as_test and not args.as_release_test
    algo = config.build()
    for i in range(stop.get(TRAINING_ITERATION, args.stop_iters)):
        results = algo.train()
        if ENV_RUNNER_RESULTS in results:
            mean_return = results[ENV_RUNNER_RESULTS].get(
                EPISODE_RETURN_MEAN, np.nan
            )
            print(f"iter={i} R={mean_return}", end="")
        if EVALUATION_RESULTS in results:
            Reval = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][
                EPISODE_RETURN_MEAN
            ]
            print(f" R(eval)={Reval}", end="")
        print()
        for key, threshold in stop.items():
            val = results
            for k in key.split("/"):
                try:
                    val = val[k]
                except KeyError:
                    val = None
                    break
            if val is not None and not np.isnan(val) and val >= threshold:
                print(f"Stop criterium ({key}={threshold}) fulfilled!")

                ray.shutdown()
                return results

    ray.shutdown()
    return results

def train(
    config: "AlgorithmConfig",
    args: Optional[argparse.Namespace] = None,
    cmd_args = None,
    scheduler=None,
    stop = None,
    enable_swanlab = False
) -> Union[ResultDict, tune.result_grid.ResultGrid]:

    ray.init(
        num_cpus=args.num_cpus or None,
        local_mode=args.local_mode,
        ignore_reinit_error=True,
    )

    enhance_base_config(config,args)
    # print(config)

    # Log results
    tune_callbacks = []
    tune_callbacks.append(CheckPointCallback())
    if enable_swanlab:
        append_swanlab(tune_callbacks,args,config,name=cmd_args.run_name)

    #progress_reporter =cli_reporter(config)
    # if args.no_tune:
    #     return train_no_tune(args, config, stop=stop)

    start_time = time.time()
    # Run the actual experiment (using Tune).
    results = tune.Tuner(
        config.algo_class,
        param_space=config,
        run_config=tune.RunConfig(
            stop=stop,
            verbose=args.verbose,
            callbacks=tune_callbacks,
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=args.checkpoint_at_end,
            ),
            #progress_reporter=progress_reporter,
        ),
        tune_config=tune.TuneConfig(
            # metric="env_runners/module_episode_returns_mean/policy_1",
            # mode="max",
            num_samples=args.num_samples,#default to 1
            max_concurrent_trials=args.max_concurrent_trials,
            scheduler=scheduler,
            trial_name_creator=trial_str_creator,
            trial_dirname_creator=trial_str_creator,
        ),
    ).fit()

    time_taken = time.time() - start_time
    print('time_taken ',str(time_taken/60))
    ray.shutdown()

    # Error out, if Tuner.fit() failed to run.
    if results.errors:
        raise RuntimeError(
            "Running the example script resulted in one or more errors! "
            f"{[e.args[0].args[2] for e in results.errors]}"
        )
    return results


def append_swanlab(tune_callbacks,args,config,name = None):
    from utils.swanlab.swanlab_logger_callback import SwanLabLoggerCallback
    # 设置环境变量，静默 wandb 输出
    project = args.swanlab_project or (
            args.algo.lower() + "-" + re.sub("\\W+", "-", str(config.env).lower())
    )
    print("swanlab with project: ", project)
    kwargs = {
        "name": name if name else args.wandb_run_name,
    }

    tune_callbacks.append(
        # WandbLoggerCallback(
        #     api_key=wandb_key,
        #     project=project,
        #     upload_checkpoints=args.upload_checkpoints_to_wandb,
        #     **kwargs,
        # )
        SwanLabLoggerCallback(
            api_key=args.swanlab_key,
            project=project,
            workspace='Eliment-li',
            upload_checkpoints=True,
            **kwargs,
        )
    )

def cli_reporter(config):
    # Force Tuner to use old progress output as the new one silently ignores our custom
    # `CLIReporter`.
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

    # Auto-configure a CLIReporter (to log the results to the console).
    # Use better ProgressReporter for multi-agent cases: List individual policy rewards.
    CLIReporter(
        metric_columns={
            **{
                TRAINING_ITERATION: "iter",
                "time_total_s": "total time (s)",
                NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
                f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "combined return",
            },
            **{
                (
                    f"{ENV_RUNNER_RESULTS}/module_episode_returns_mean/" f"{pid}"
                ): f"return {pid}"
                for pid in config.policies
            },
        },
    )

def trial_str_creator(trial):
    return "{}_{}".format(trial.trainable_name, trial.trial_id)

if __name__ == '__main__':
    pass