import os

from gymnasium import register
from hydra import initialize, compose
from omegaconf import OmegaConf
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module import RLModule

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

    print("Training completed. Restoring new RLModule for action inference.")
    # Get the last checkpoint from the above training run.
    # best_result = results.get_best_result(metric='env_runners/episode_reward_mean', mode='max').checkpoint
    #
    # best_path = best_result.to_directory()
    best_path = r'C:\Users\ADMINI~1\AppData\Local\Temp\checkpoint_tmp_ecdba2f2107445bba129010d9834a024'
    # Create new RLModule and restore its state from the last algo checkpoint.
    # Note that the checkpoint for the RLModule can be found deeper inside the algo
    # checkpoint's subdirectories ([algo dir] -> "learner/" -> "module_state/" ->
    # "[module ID]):
    rl_module = RLModule.from_checkpoint(
        os.path.join(
            best_path,
            "learner_group",
            "learner",
            "rl_module",
            'policy_1',
        )
    )

    # Create an env to do inference in.
    #env = gym.make(args.env,disable_env_checker = True)
    env = Env_0()
    obs, info = env.reset()

    num_episodes = 0
    episode_return = 0.0

    while num_episodes < 10:
        obs = obs['agent_1']
        # Compute an action using a B=1 observation "batch".
        input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0)}
        # No exploration.
        if not args.explore_during_inference:
            rl_module_out = rl_module.forward_inference(input_dict)
        # Using exploration.
        else:
            rl_module_out = rl_module.forward_exploration(input_dict)

        # For discrete action spaces used here, normally, an RLModule "only"
        # produces action logits, from which we then have to sample.
        # However, you can also write custom RLModules that output actions
        # directly, performing the sampling step already inside their
        # `forward_...()` methods.
        logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
        # Perform the sampling step in numpy for simplicity.
        action = np.random.choice(env.action_space.n, p=softmax(logits[0]))
        # Send the computed action `a` to the env.
        obs, reward, terminated, truncated, _ = env.step(action)
        obs = obs['agent_1']
        episode_return += reward
        # Is the episode `done`? -> Reset.
        if terminated or truncated:
            print(f"Episode done: Total reward = {episode_return}")
            obs, info = env.reset()
            num_episodes += 1
            episode_return = 0.0

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
    inference(base_config,args,None)