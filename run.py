import os

from gymnasium import register
from hydra import initialize, compose
from omegaconf import OmegaConf
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module import RLModule
from sympy import pprint

from config import ConfigSingleton
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

from envs.env_0 import Env_0

def inference(base_config, args, results):

    #base_config = get_trainable_cls(args.algo).get_default_config()
    # Get the last checkpoint from the above training run.
    # best_result = results.get_best_result(metric='env_runners/episode_reward_mean', mode='max').checkpoint
    #
    # best_path = best_result.to_directory()
    best_path = r'd:\checkpoint'
    # Create new RLModule and restore its state from the last algo checkpoint.
    # Note that the checkpoint for the RLModule can be found deeper inside the algo
    # checkpoint's subdirectories ([algo dir] -> "learner/" -> "module_state/" ->
    # "[module ID]):
    # rl_module = RLModule.from_checkpoint(
    #     os.path.join(
    #         best_path,
    #         "learner_group",
    #         "learner",
    #         "rl_module",
    #         'policy_1',
    #     )
    # )

    if  not isinstance(results, str):
        checkpoint = results.get_best_result(metric='env_runners/episode_reward_mean', mode='max').checkpoint
        checkpoint = checkpoint.to_directory()
        print(f'best checkpoint: {checkpoint}')
        algo = Algorithm.from_checkpoint(path=checkpoint)
    else:
        algo = Algorithm.from_checkpoint(path=results)


    # Create an env to do inference in.
    #env = gym.make(args.env,disable_env_checker = True)
    env = Env_0()
    obs, info = env.reset()

    num_episodes = 0
    episode_return = 0.0

    while num_episodes < 10:
        obs1 = obs['agent_1']
        a = algo.compute_single_action(
            observation=obs1,
            explore=None,
            policy_id="policy_1",  # <- default value
        )
        num_episodes += 1
        obs, reward, done,terminated, truncated, info = env.step(a)
        # if terminated or truncated:
        #     print(f"Episode done: Total reward = {episode_return}")
        #     obs, info = env.reset()
        #     num_episodes += 1
        #     episode_return = 0.0

    print(f"Done performing action inference through {num_episodes} Episodes")
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
    inference(base_config,args,r'd:/checkpoint')