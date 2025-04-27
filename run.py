import os

from gymnasium import register
from hydra import initialize, compose
from omegaconf import OmegaConf
from ray.rllib.connectors.common.module_to_agent_unmapping import ModuleToAgentUnmapping
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module import RLModule, MultiRLModule
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.examples.envs.classes.stateless_cartpole import StatelessCartPole
from sympy import pprint
import os

from ray.rllib.connectors.env_to_module import EnvToModulePipeline
from ray.rllib.connectors.module_to_env import ModuleToEnvPipeline
from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_ENV_TO_MODULE_CONNECTOR,
    COMPONENT_MODULE_TO_ENV_CONNECTOR,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
    DEFAULT_MODULE_ID,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.examples.envs.classes.stateless_cartpole import StatelessCartPole
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env
from torch.distributed.pipelining import pipeline

from config import ConfigSingleton
from utils.evaluate import plot_reward
from utils.run_helper import train, evaluate
import gymnasium as gym
import numpy as np
import os
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy, softmax
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
torch, _ = try_import_torch()
def policy_mapping_fn(agent_id, episode, **kwargs):
    """
    Map agent IDs to policy names based on their numeric suffix.

    Args:
        agent_id (str): The ID of the agent (e.g., "agent_1").
        episode (object): The current episode object (not used here).
        **kwargs: Additional keyword arguments (not used here).

    Returns:
        str: The name of the policy corresponding to the agent ID (e.g., "policy_1").
    """
    # Extract the numeric suffix from the agent ID and map it to "policy_n"
    try:
        agent_number = agent_id.split('_')[1]  # Split and get the number part
        return f"policy_{agent_number}"       # Return the mapped policy name
    except IndexError:
        raise ValueError(f"Invalid agent_id format: {agent_id}. Expected 'agent_n'.")


from ray.rllib.connectors.env_to_module.flatten_observations import FlattenObservations
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env  # noqa
def new_env():
    return  Env_0()
from envs.env_0 import Env_0
register_env("Env_0", new_env)
def inference(base_config, args, results):

    if isinstance(results,str):
        best_path = results
    else:
        best_result = results.get_best_result(metric='env_runners/episode_reward_mean', mode='max').checkpoint
        best_path = best_result.to_directory()
        print('best_path=', best_path)
    # Create the env.
    env = Env_0()

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
    stop_timesteps = args.stop_timesteps
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

        # No exploration.
        # if not args.explore_during_inference:
        #     rl_module_out = rl_module.forward_inference(input_dict)
        # # Using exploration.
        # else:
        #     rl_module_out = rl_module.forward_exploration(input_dict)

        new_input = {}
        for i, tensor in enumerate(input_dict['default_policy']['obs']):
            key = f'policy_{i+1}'
            new_input[key] = {'obs':tensor}

        module_out = rl_module._forward_inference(new_input)
        #Using exploration.
        # rl_module_out = rl_module.forward_exploration(input_dict)

        # new_out  = {}
        # i = 1
        # print('###',module_out.keys())
        # for key in module_out.keys():
        #     new_out[f'policy_{i}'] = module_out[key]
        #     i +=1
        # # module_to_env 定义或使用的有问题，导致后续报错
        # print(new_out)
        print(module_out)
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
        actions = {
            'agent_1':to_env['policy_1']['actions'],
            'agent_2':to_env['policy_2']['actions']
        }
        #action = to_env.pop(Columns.ACTIONS)[0]

        obs, reward, terminated, truncated, _ = env.step(actions)
        rewrads.append(reward['agent_1'])
        # print(f'####obs {max_steps} ###\n')
        # print(obs['agent_1'])
        # print(f'####reward{max_steps} ###\n')
        # print(reward)

        # Keep our `Episode` instance updated at all times.
        # update_episode()
        stop_timesteps -= 1
        if  terminated['__all__'] or truncated or stop_timesteps <= 0:
            print(f'{terminated},{truncated},{stop_timesteps}')
            break
    print(rewrads)
    plot_reward([rewrads])


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

if __name__ == "__main__":
    # ConfigSingleton().add('num_agents' ,2)
    args = ConfigSingleton().get_args()

    # You can also register the envs creator function explicitly with:
    # register_env("envs", lambda cfg: RockPaperScissors({"sheldon_cooper_mode": False}))

    # Or you can hard code certain settings into the Env's constructor (`config`).
    # register_env(
    #    "rock-paper-scissors-w-sheldon-mode-activated",
    #    lambda config: RockPaperScissors({**config, **{"sheldon_cooper_mode": True}}),
    # )

    # Or allow the RLlib user to set more c'tor options via their algo config:
    # config.environment(env_config={[c'tor arg name]: [value]})
    # register_env("rock-paper-scissors", lambda cfg: RockPaperScissors(cfg))
    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            Env_0,
            env_config={"key": "value"},
        )
        .env_runners(
            env_to_module_connector=lambda env: FlattenObservations(multi_agent=True),
        )
        .multi_agent(
            # Define two policies.
            policies={"policy_1", "policy_2"},
            # Map agent "player1" to policy "player1" and agent "player2" to policy
            # "player2".
            policy_mapping_fn=policy_mapping_fn
        )
    # .rl_module(
    #     rl_module_spec=MultiRLModuleSpec(rl_module_specs={
    #         "learning_policy": RLModuleSpec(),
    #         "random_policy": RLModuleSpec(rl_module_class=RandomRLModule),
    #     }),
    # )
    )

    #results = train(base_config, args)
    inference(base_config,args,r'C:\Users\90471\AppData\Local\Temp\checkpoint_tmp_cd5c59fcbcf14af0938ac85326156ca6')
    #inference(base_config,args,r'd:/checkpoint')