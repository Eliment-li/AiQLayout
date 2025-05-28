import argparse
import json
import logging
import os
import pprint
import random
import re
import time
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import gymnasium as gym
from gymnasium import register
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Tuple as GymTuple
import numpy as np
import tree  # pip install dm_tree

import ray
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback, WANDB_ENV_VAR
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
    DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY,
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    NUM_ENV_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.typing import ResultDict
from ray.tune import CLIReporter
from ray.tune.result import TRAINING_ITERATION

from config import ConfigSingleton, enhance_base_config
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.offline.dataset_reader import DatasetReader

from envs.env_0 import Env_0
from utils.checkpoint import CheckPointCallback

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
    tune_callbacks: Optional[List] = None,
    scheduler=None,
    stop = None,
    enable_wandb = False
) -> Union[ResultDict, tune.result_grid.ResultGrid]:
    """Given an algorithm config and some command line args, runs an experiment.

    The function sets up an Algorithm object from the given config (altered by the
    contents of `args`), then runs the Algorithm via Tune (or manually, if
    `args.no_tune` is set to True) using the stopping criteria in `stop`.

    At the end of the experiment, if `args.as_test` is True, checks, whether the
    Algorithm reached the `success_metric` (if None, use `env_runners/
    episode_return_mean` with a minimum value of `args.stop_reward`).

    See https://github.com/ray-project/ray/tree/master/rllib/examples for an overview
    of all supported command line options.

    Args:
        config: The AlgorithmConfig object to use for this experiment. This base
            config will be automatically "extended" based on some of the provided
            `args`. For example, `args.num_env_runners` is used to set
            `config.num_env_runners`, etc..
        args: A config, It must have the following
            properties defined: `stop_iters`, `stop_reward`, `stop_timesteps`,
            `no_tune`, `verbose`, `checkpoint_freq`, `as_test`. Optionally, for WandB
            logging: `wandb_key`, `wandb_project`, `wandb_run_name`.

        tune_callbacks: A list of Tune callbacks to configure with the tune.Tuner.
            In case `args.wandb_key` is provided, appends a WandB logger to this
            list
    Returns:
        The last ResultDict from a --no-tune run OR the tune.Tuner.fit()
        results.
    """
    ray.init(
        num_cpus=args.num_cpus or None,
        local_mode=args.local_mode,
        ignore_reinit_error=True,
    )

    enhance_base_config(config,args)
    #print(config)

    # Log results using WandB.
    tune_callbacks = tune_callbacks or []
    # tune_callbacks.append(CheckPointCallback())
    if enable_wandb:
        append_wandb(tune_callbacks,args,config,name=cmd_args.run_name,group = cmd_args.wandb_group)

    progress_reporter =cli_reporter(config)
    if args.no_tune:
        return train_no_tune(args, config, stop=stop)

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


def append_wandb(tune_callbacks,args,config,group,name = None):
    wandb_key = args.wandb_key
    # 设置环境变量，静默 wandb 输出
    os.environ["WANDB_SILENT"] = "true"
    project = args.wandb_project or (
            args.algo.lower() + "-" + re.sub("\\W+", "-", str(config.env).lower())
    )
    kwargs = {
        "name": name if name else args.wandb_run_name,
        "group":group
        #"silent": True
    }
    tune_callbacks.append(
        WandbLoggerCallback(
            api_key=wandb_key,
            project=project,
            upload_checkpoints=args.upload_checkpoints_to_wandb,
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
def register_env():
    for version in range(1):
        register(
            id='Env_'+str(version),
            # entry_point='core.envs.circuit_env:CircuitEnv',
            entry_point='envs.env_'+str(version)+':Env_'+str(version),
            #max_episode_steps=999999,
        )




if __name__ == '__main__':
    pass