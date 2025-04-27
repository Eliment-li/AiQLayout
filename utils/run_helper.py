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

def train(
    base_config: "AlgorithmConfig",
    args: Optional[argparse.Namespace] = None,
    *,
    stop: Optional[Dict] = None,
    trainable: Optional[Type] = None,
    tune_callbacks: Optional[List] = None,
    keep_config: bool = False,
    scheduler=None,
    progress_reporter=None,
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
        base_config: The AlgorithmConfig object to use for this experiment. This base
            config will be automatically "extended" based on some of the provided
            `args`. For example, `args.num_env_runners` is used to set
            `config.num_env_runners`, etc..
        args: A config, It must have the following
            properties defined: `stop_iters`, `stop_reward`, `stop_timesteps`,
            `no_tune`, `verbose`, `checkpoint_freq`, `as_test`. Optionally, for WandB
            logging: `wandb_key`, `wandb_project`, `wandb_run_name`.
        stop: An optional dict mapping ResultDict key strings (using "/" in case of
            nesting, e.g. "env_runners/episode_return_mean" for referring to
            `result_dict['env_runners']['episode_return_mean']` to minimum
            values, reaching of which will stop the experiment). Default is:
            {
            "env_runners/episode_return_mean": args.stop_reward,
            "training_iteration": args.stop_iters,
            "num_env_steps_sampled_lifetime": args.stop_timesteps,
            }

        trainable: The Trainable sub-class to run in the tune.Tuner. If None (default),
            use the registered RLlib Algorithm class specified by args.algo.
        tune_callbacks: A list of Tune callbacks to configure with the tune.Tuner.
            In case `args.wandb_key` is provided, appends a WandB logger to this
            list.
        keep_config: Set this to True, if you don't want this utility to change the
            given `base_config` in any way and leave it as-is. This is helpful
            for those example scripts which demonstrate how to set config settings
            that are otherwise taken care of automatically in this function (e.g.
            `num_env_runners`).

    Returns:
        The last ResultDict from a --no-tune run OR the tune.Tuner.fit()
        results.
    """

    # If run --as-release-test, --as-test must also be set.
    if args.as_release_test:
        args.as_test = True

    # Initialize Ray.
    ray.init(
        num_cpus=args.num_cpus or None,
        local_mode=args.local_mode,
        ignore_reinit_error=True,
    )

    # Define one or more stopping criteria.
    if stop is None:
        stop = {
            # the results does not contail the metric EPISODE_RETURN_MEAN
            # f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
            f"{ENV_RUNNER_RESULTS}/{NUM_ENV_STEPS_SAMPLED_LIFETIME}": (
                args.stop_timesteps
            ),
            TRAINING_ITERATION: args.stop_iters,
        }

    config = base_config

    # Enhance the `base_config`, based on provided `args`.
    if not keep_config:
        enhance_config(config,args)

    # Run the experiment w/o Tune (directly operate on the RLlib Algorithm object).
    if args.no_tune:
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

    # Run the experiment using Ray Tune.

    # Log results using WandB.
    tune_callbacks = tune_callbacks or []
    # if hasattr(args, "wandb_key") and (
    #     args.wandb_key is not None or WANDB_ENV_VAR in os.environ
    # ):
    #     append_wandb(tune_callbacks,args,config)


    if progress_reporter is None and args.num_agents > 0:
        progress_reporter =cli_reporter(config)



    # Run the actual experiment (using Tune).
    start_time = time.time()
    results = tune.Tuner(
        trainable or config.algo_class,
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
    print('time_taken=',str(time_taken/60))

    ray.shutdown()

    # Error out, if Tuner.fit() failed to run.
    if results.errors:
        raise RuntimeError(
            "Running the example script resulted in one or more errors! "
            f"{[e.args[0].args[2] for e in results.errors]}"
        )


    return results

def enhance_config(config,args):
    # Set the framework.
    config.framework(args.framework)

    # Add an env specifier (only if not already set in config)?
    if args.env is not None and config.env is None:
        config.environment(args.env)

    # Disable the new API stack?
    # @see https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.api_stack.html#ray-rllib-algorithms-algorithm-config-algorithmconfig-api-stack
    if not args.enable_new_api_stack:
        config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )

    # Define EnvRunner scaling and behavior.
    if args.num_env_runners is not None:
        config.env_runners(num_env_runners=args.num_env_runners)
    if args.num_envs_per_env_runner is not None:
        config.env_runners(num_envs_per_env_runner=args.num_envs_per_env_runner)

    # Define compute resources used automatically (only using the --num-learners
    # and --num-gpus-per-learner args).
    # New stack.
    if config.enable_rl_module_and_learner:
        if args.num_gpus is not None and args.num_gpus > 0:
            raise ValueError(
                "--num-gpus is not supported on the new API stack! To train on "
                "GPUs, use the command line options `--num-gpus-per-learner=1` and "
                "`--num-learners=[your number of available GPUs]`, instead."
            )

        # Do we have GPUs available in the cluster?
        num_gpus_available = ray.cluster_resources().get("GPU", 0)
        # Number of actual Learner instances (including the local Learner if
        # `num_learners=0`).
        num_actual_learners = (
                                  args.num_learners
                                  if args.num_learners is not None
                                  else config.num_learners
                              ) or 1  # 1: There is always a local Learner, if num_learners=0.
        # How many were hard-requested by the user
        # (through explicit `--num-gpus-per-learner >= 1`).
        num_gpus_requested = (args.num_gpus_per_learner or 0) * num_actual_learners
        # Number of GPUs needed, if `num_gpus_per_learner=None` (auto).
        num_gpus_needed_if_available = (
                                           args.num_gpus_per_learner
                                           if args.num_gpus_per_learner is not None
                                           else 1
                                       ) * num_actual_learners
        # Define compute resources used.
        config.resources(num_gpus=0)  # old API stack setting
        if args.num_learners is not None:
            config.learners(num_learners=args.num_learners)

        # User wants to use aggregator actors per Learner.
        if args.num_aggregator_actors_per_learner is not None:
            config.learners(
                num_aggregator_actors_per_learner=(
                    args.num_aggregator_actors_per_learner
                )
            )

        # User wants to use GPUs if available, but doesn't hard-require them.
        if args.num_gpus_per_learner is None:
            if num_gpus_available >= num_gpus_needed_if_available:
                config.learners(num_gpus_per_learner=1)
            else:
                config.learners(num_gpus_per_learner=0)
        # User hard-requires n GPUs, but they are not available -> Error.
        elif num_gpus_available < num_gpus_requested:
            raise ValueError(
                "You are running your script with --num-learners="
                f"{args.num_learners} and --num-gpus-per-learner="
                f"{args.num_gpus_per_learner}, but your cluster only has "
                f"{num_gpus_available} GPUs!"
            )

        # All required GPUs are available -> Use them.
        else:
            config.learners(num_gpus_per_learner=args.num_gpus_per_learner)

        # Set CPUs per Learner.
        if args.num_cpus_per_learner is not None:
            config.learners(num_cpus_per_learner=args.num_cpus_per_learner)

    # Old stack (override only if arg was provided by user).
    elif args.num_gpus is not None:
        config.resources(num_gpus=args.num_gpus)

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

def evaluate(results):
    args = ConfigSingleton().get_args()
    register_env()
    if  not isinstance(results, str):
        checkpoint = results.get_best_result(metric='env_runners/episode_reward_mean', mode='max').checkpoint
        checkpoint = checkpoint.to_directory()
        print(f'best checkpoint: {checkpoint}')
        algo = Algorithm.from_checkpoint(path=checkpoint)
    else:
        algo = Algorithm.from_checkpoint(path=results)

    env = Env_0()
    obs, info = env.reset()

    #####################

    from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
        DefaultPPOTorchRLModule
    )
    from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog

    # Create an instance of the default RLModule used by PPO.
    module = algo.get_module('policy_1')
    action_dist_class = module.get_inference_action_dist_cls()
    terminated = False
    while not terminated:
        fwd_ins = {"obs": torch.Tensor([obs])}
        fwd_outputs = module.forward_inference(fwd_ins)
        # this can be either deterministic or stochastic distribution
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = action_dist.sample()[0].numpy()
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)

    #######################
    # for i in range(10):
    #     obs, reward, done, truncated, info = env.step(action)
    #     #trace
    #
    #     print('done = %r, action = %r, reward = %r,  info = %r \n' % (done,a, reward,info['occupy']))
    #     episode_reward *=args.gamma
    #     episode_reward += reward

        # if done:
        #     print('env done = %r, action = %r, reward = %r  occupy =  {%r} ' % (done,a, reward, info['occupy']))
        #     print(f"Episode done: Total reward = {episode_reward}")
        #     break

    algo.stop()

    trace = []
    # if args.show_trace:
    #     show_trace(trace.transpose())




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
    evaluate(r'C:\Users\ADMINI~1\AppData\Local\Temp\checkpoint_tmp_ecdba2f2107445bba129010d9834a024')