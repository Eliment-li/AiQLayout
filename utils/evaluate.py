import os
import argparse
import json
import logging
import os
from pprint import pprint
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

from ray.rllib.utils.metrics import (
    DIFF_NUM_GRAD_UPDATES_VS_SAMPLER_POLICY,
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    NUM_ENV_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)

from ray.tune.result import TRAINING_ITERATION

from ray.rllib.connectors.env_to_module import EnvToModulePipeline
from ray.rllib.connectors.module_to_env import ModuleToEnvPipeline
from ray.rllib.core import COMPONENT_ENV_RUNNER, COMPONENT_ENV_TO_MODULE_CONNECTOR, COMPONENT_LEARNER_GROUP, \
    COMPONENT_LEARNER, COMPONENT_RL_MODULE, COMPONENT_MODULE_TO_ENV_CONNECTOR
from ray.rllib.core.rl_module import MultiRLModule
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

from envs.env_0 import Env_0
from envs.env_1 import Env_1
from results.plot_results import plot_reward


def evaluate_v2(base_config, args, results):
    if isinstance(results, str):
        best_path = results
    else:
        best_result = results.get_best_result(metric='env_runners/episode_reward_mean', mode='max').checkpoint
        best_path = best_result.to_directory()
        print('best_path=', best_path)
    # Create the env.
    env = Env_1()

    # Create the env-to-module pipeline from the checkpoint.
    print("Restore env-to-module connector from checkpoint ...", end="")
    env_to_module = EnvToModulePipeline.from_checkpoint(
        os.path.join(
            best_path,
            COMPONENT_ENV_RUNNER,
            COMPONENT_ENV_TO_MODULE_CONNECTOR,
        )
    )
    print(" ok")

    print("Restore RLModule from checkpoint ...", end="")
    rl_module = MultiRLModule.from_checkpoint(
        os.path.join(
            best_path,
            COMPONENT_LEARNER_GROUP,
            COMPONENT_LEARNER,
            COMPONENT_RL_MODULE,

            # 'policy_1' # DEFAULT_MODULE_ID,
        )
    )
    print(" ok")

    # For the module-to-env pipeline, we will use the convenient config utility.
    print("Restore module-to-env connector from checkpoint ...", end="")
    # This class does nothing, need fix, see EnvToModulePipeline
    module_to_env = ModuleToEnvPipeline.from_checkpoint(
        os.path.join(
            best_path,
            COMPONENT_ENV_RUNNER,
            COMPONENT_MODULE_TO_ENV_CONNECTOR,
        )
    )
    # remove default pipeline that incompatible with multi-agent case
    incompatible_pipelines = [
        'UnBatchToIndividualItems',
        'ModuleToAgentUnmapping',
        'RemoveSingleTsTimeRankFromBatch',
        'NormalizeAndClipActions',
        'ListifyDataForVectorEnv',
        'ModuleToEnvPipeline',
    ]
    for pipeline in incompatible_pipelines:
        module_to_env.remove(pipeline)

    rewrads = [[] for i in range(env.num_qubits)]
    distance = [[] for i in range(env.num_qubits)]

    obs, _ = env.reset()
    terminated, truncated = False, False
    stop_timesteps = 20
    while True:

        shared_data = {}
        episode = MultiAgentEpisode(
            observations=[obs],
            observation_space=env.observation_spaces,
            action_space=env.action_spaces,
        )
        input_dict = env_to_module(
            episodes=[episode],  # ConnectorV2 pipelines operate on lists of episodes.
            rl_module=rl_module,
            explore=args.explore_during_inference,
            shared_data=shared_data,
        )

        new_input = {}
        obs = input_dict['default_policy']['obs'][0]
        new_input[ f'policy_{env.player_now}'] = {'obs': obs}

        # No exploration.
        module_out = rl_module._forward_inference(new_input)
        # Using exploration.
        # rl_module_out = rl_module.forward_exploration(input_dict)
        to_env = module_to_env(
            batch=module_out,
            episodes=[episode],  # ConnectorV2 pipelines operate on lists of episodes.
            rl_module=rl_module,
            explore=args.explore_during_inference,
            shared_data=shared_data,
        )

        # Send the computed action to the env. Note that the RLModule and the
        # connector pipelines work on batched data (B=1 in this case), whereas the Env
        # is not vectorized here, so we need to use `action[0]`.
        # actions = {
        #     'agent_1':to_env['policy_1']['actions'],
        #     'agent_2':to_env['policy_2']['actions']
        # }
        #actions = {f'agent_{i + 1}': to_env[f'policy_{i + 1}']['actions'] for i in range(len(to_env))}
        actions = {f'agent_{env.player_now}': to_env[f'policy_{env.player_now}']['actions']}

        last_player = env.player_now
        obs, reward, terminated, truncated, info = env.step(actions)
        rewrads.append(reward['agent_1'])
        rewrads[last_player - 1].append(reward)

        distance[last_player - 1].append(info[f'agent_{last_player}']['distance'])
        # Keep our `Episode` instance updated at all times.
        # update_episode()
        stop_timesteps -= 1
        if terminated['__all__'] or truncated or stop_timesteps <= 0:
            pprint(obs)
            # print(f'{terminated},{truncated},{stop_timesteps}')
            break

    # print(rewrads)
    plot_reward([rewrads, distance])
    print(env.chip.position)
    print(env.chip.state)

def evaluate(base_config, args, results):

    if isinstance(results,str):
        best_path = results
    else:
        best_result = results.get_best_result(metric='env_runners/episode_reward_mean', mode='max').checkpoint
        best_path = best_result.to_directory()
        print('best_path=', best_path)
    # Create the env.
    env = Env_1()

    # Create the env-to-module pipeline from the checkpoint.
    print("Restore env-to-module connector from checkpoint ...", end="")
    env_to_module = EnvToModulePipeline.from_checkpoint(
        os.path.join(
            best_path,
            COMPONENT_ENV_RUNNER,
            COMPONENT_ENV_TO_MODULE_CONNECTOR,
        )
    )
    print(" ok")

    print("Restore RLModule from checkpoint ...", end="")
    rl_module = MultiRLModule.from_checkpoint(
        os.path.join(
            best_path,
            COMPONENT_LEARNER_GROUP,
            COMPONENT_LEARNER,
            COMPONENT_RL_MODULE,

           # 'policy_1' # DEFAULT_MODULE_ID,
        )
    )
    print(" ok")

    # For the module-to-env pipeline, we will use the convenient config utility.
    print("Restore module-to-env connector from checkpoint ...", end="")
    #This class does nothing, need fix, see EnvToModulePipeline
    module_to_env = ModuleToEnvPipeline.from_checkpoint(
        os.path.join(
            best_path,
            COMPONENT_ENV_RUNNER,
            COMPONENT_MODULE_TO_ENV_CONNECTOR,
        )
    )
    #remove default pipeline that incompatible with multi-agent case
    incompatible_pipelines = [
        'UnBatchToIndividualItems',
        'ModuleToAgentUnmapping',
        'RemoveSingleTsTimeRankFromBatch',
        'NormalizeAndClipActions',
        'ListifyDataForVectorEnv',
        'ModuleToEnvPipeline',
    ]
    for pipeline in incompatible_pipelines:
        module_to_env.remove(pipeline)

    rewrads = []
    distance = []

    obs, _ = env.reset()
    terminated, truncated = False, False
    stop_timesteps = 20
    while True:

        shared_data = {}
        episode = MultiAgentEpisode(
            observations=[obs],
            observation_space=env.observation_spaces,
            action_space=env.action_spaces,
        )
        input_dict = env_to_module(
            episodes=[episode],  # ConnectorV2 pipelines operate on lists of episodes.
            rl_module=rl_module,
            explore=args.explore_during_inference,
            shared_data=shared_data,
        )

        new_input = {}
        for i, tensor in enumerate(input_dict['default_policy']['obs']):
            key = f'policy_{i+1}'
            new_input[key] = {'obs':tensor}
        # No exploration.
        module_out = rl_module._forward_inference(new_input)
        #Using exploration.
        # rl_module_out = rl_module.forward_exploration(input_dict)
        to_env = module_to_env(
            batch=module_out,
            episodes=[episode],  # ConnectorV2 pipelines operate on lists of episodes.
            rl_module=rl_module,
            explore=args.explore_during_inference,
            shared_data=shared_data,
        )

        # Send the computed action to the env. Note that the RLModule and the
        # connector pipelines work on batched data (B=1 in this case), whereas the Env
        # is not vectorized here, so we need to use `action[0]`.
        # actions = {
        #     'agent_1':to_env['policy_1']['actions'],
        #     'agent_2':to_env['policy_2']['actions']
        # }
        actions = {f'agent_{i+1}':to_env[f'policy_{i+1}']['actions'] for i in range(len(to_env))}

        obs, reward, terminated, truncated, info = env.step(actions)
        rewrads.append(reward['agent_1'])
        distance.append(info['agent_1']['distance'])
        # Keep our `Episode` instance updated at all times.
        # update_episode()
        stop_timesteps -= 1
        if  terminated['__all__'] or truncated or stop_timesteps <= 0:
            pprint(obs)
            #print(f'{terminated},{truncated},{stop_timesteps}')
            break

    # print(rewrads)
    plot_reward([rewrads,distance])
    print(env.chip.position)
    print(env.chip.state)


def update_episode(episode,obs,action,rewrad, terminated,truncated,to_env):
    # Keep our `Episode` instance updated at all times.
    episode.add_env_step(
        obs,
        action,
        rewrad,
        terminated=terminated,
        truncated=truncated,
        # Same here: [0] b/c RLModule output is batched (w/ B=1).
        extra_model_outputs={k: v[0] for k, v in to_env.items()},
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