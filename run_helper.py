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

from config import ConfigSingleton
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.offline.dataset_reader import DatasetReader

from envs.env_0 import Env_0

jax, _ = try_import_jax()
tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

logger = logging.getLogger(__name__)


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
    tune_callbacks: Optional[List] = None,
    scheduler=None,
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

    stop = {
            # the results does not contail the metric EPISODE_RETURN_MEAN
            # f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
            f"{ENV_RUNNER_RESULTS}/{NUM_ENV_STEPS_SAMPLED_LIFETIME}": (
                args.stop_timesteps
            ),
            TRAINING_ITERATION: args.stop_iters,
    }

    enhance_config(config,args)
    print(config)

    # Run the experiment using Ray Tune.
    # Log results using WandB.
    tune_callbacks = tune_callbacks or []
    # if hasattr(args, "wandb_key") and (
    #     args.wandb_key is not None or WANDB_ENV_VAR in os.environ
    # ):
    #     append_wandb(tune_callbacks,args,config)

    progress_reporter =cli_reporter(config)
    if args.no_tune:
        return train_no_tune(args, config, stop=stop)

    # Run the actual experiment (using Tune).
    start_time = time.time()
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
            progress_reporter=progress_reporter,
        ),
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,#default to 1
            max_concurrent_trials=args.max_concurrent_trials,
            scheduler=scheduler,
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

def enhance_config(config,args):
    config.framework(args.framework)

    # Disable the new API stack
    # @see https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.api_stack.html#ray-rllib-algorithms-algorithm-config-algorithmconfig-api-stack

    if args.num_env_runners is not None:
        config.env_runners(num_env_runners=args.num_env_runners)
    if args.num_envs_per_env_runner is not None:
        config.env_runners(num_envs_per_env_runner=args.num_envs_per_env_runner)

    # New stack.
    if args.enable_new_api_stack:
        # GPUs available in the cluster?
        num_gpus_available = ray.cluster_resources().get("GPU", 0)
        num_gpus_requested = (args.num_gpus_per_learner or 0) * args.num_learners

        # Define compute resources used.
        #config.resources(num_gpus=num_gpus_available)  # old API stack setting

        #set num_learners and num_gpus_per_learner
        config.learners(num_learners=args.num_learners)
        if num_gpus_available >= num_gpus_requested:
            # All required GPUs are available -> Use them.
            config.learners(num_gpus_per_learner=args.num_gpus_per_learner)
        else:
            raise ValueError(
                "You are running your script with --num-learners="
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
    # if args.evaluation_interval > 0:
    #     config.evaluation(
    #         evaluation_num_env_runners=args.evaluation_num_env_runners,
    #         evaluation_interval=args.evaluation_interval,
    #         evaluation_duration=args.evaluation_duration,
    #         evaluation_duration_unit=args.evaluation_duration_unit,
    #         evaluation_parallel_to_training=args.evaluation_parallel_to_training,
    #     )

    # Set the log-level (if applicable).
    if args.log_level is not None:
        config.debugging(log_level=args.log_level)

    # Set the output dir (if applicable).
    if args.output is not None:
        config.offline_data(output=args.output)

def append_wandb(tune_callbacks,args,config):
    wandb_key = args.wandb_key or os.environ[WANDB_ENV_VAR]
    project = args.wandb_project or (
            args.algo.lower() + "-" + re.sub("\\W+", "-", str(config.env).lower())
    )
    tune_callbacks.append(
        WandbLoggerCallback(
            api_key=wandb_key,
            project=project,
            upload_checkpoints=True,
            **({"name": args.wandb_run_name} if args.wandb_run_name else {}),
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

def register_env():
    for version in range(1):
        register(
            id='Env_'+str(version),
            # entry_point='core.envs.circuit_env:CircuitEnv',
            entry_point='envs.env_'+str(version)+':Env_'+str(version),
            #max_episode_steps=999999,
        )


def test_trail(results,args, success_metric: Optional[Dict] = None,stop=None):
    '''
          success_metric: Only relevant if `args.as_test` is True.
            A dict mapping a single(!) ResultDict key string (using "/" in
            case of nesting, e.g. "env_runners/episode_return_mean" for referring
            to `result_dict['env_runners']['episode_return_mean']` to a single(!)
            minimum value to be reached in order for the experiment to count as
            successful. If `args.as_test` is True AND this `success_metric` is not
            reached with the bounds defined by `stop`, will raise an Exception.

            stop: An optional dict mapping ResultDict key strings (using "/" in case of
            nesting, e.g. "env_runners/episode_return_mean" for referring to
            `result_dict['env_runners']['episode_return_mean']` to minimum
            values, reaching of which will stop the experiment). Default is:
            {
            "env_runners/episode_return_mean": args.stop_reward,
            "training_iteration": args.stop_iters,
            "num_env_steps_sampled_lifetime": args.stop_timesteps,
            }
    '''
    if stop is None:
        stop = {
            f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
            f"{ENV_RUNNER_RESULTS}/{NUM_ENV_STEPS_SAMPLED_LIFETIME}": (
                args.stop_timesteps
            ),
            TRAINING_ITERATION: args.stop_iters,
        }
    # If run as a test, check whether we reached the specified success criteria.
    test_passed = False
    if args.as_test:
        # Success metric not provided, try extracting it from `stop`.
        if success_metric is None:
            for try_it in [
                f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
                f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
            ]:
                if try_it in stop:
                    success_metric = {try_it: stop[try_it]}
                    break
            if success_metric is None:
                success_metric = {
                    f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
                }
        # TODO (sven): Make this work for more than one metric (AND-logic?).
        # Get maximum value of `metric` over all trials
        # (check if at least one trial achieved some learning, not just the final one).
        success_metric_key, success_metric_value = next(iter(success_metric.items()))
        best_value = max(
            row[success_metric_key] for _, row in results.get_dataframe().iterrows()
        )
        if best_value >= success_metric_value:
            test_passed = True
            print(f"`{success_metric_key}` of {success_metric_value} reached! ok")

        if args.as_release_test:
            trial = results._experiment_analysis.trials[0]
            stats = trial.last_result
            stats.pop("config", None)
            json_summary = {
                # "time_taken": float(time_taken),
                "trial_states": [trial.status],
                "last_update": float(time.time()),
                "stats": stats,
                "passed": [test_passed],
                "not_passed": [not test_passed],
                "failures": {str(trial): 1} if not test_passed else {},
            }
            with open(
                os.environ.get("TEST_OUTPUT_JSON", "/tmp/learning_test.json"),
                "wt",
            ) as f:
                try:
                    json.dump(json_summary, f)
                # Something went wrong writing json. Try again w/ simplified stats.
                except Exception:
                    from ray.rllib.algorithms.algorithm import Algorithm

                    simplified_stats = {
                        k: stats[k] for k in Algorithm._progress_metrics if k in stats
                    }
                    json_summary["stats"] = simplified_stats
                    json.dump(json_summary, f)

        if not test_passed:
            raise ValueError(
                f"`{success_metric_key}` of {success_metric_value} not reached!"
            )

if __name__ == '__main__':
    pass