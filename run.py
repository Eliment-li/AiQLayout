from hydra import initialize, compose
from omegaconf import OmegaConf


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

args = None
# 手动初始化 Hydra
with initialize(config_path="conf", job_name="config",version_base="1.2"):
    # 加载配置文件
    confs = compose(config_name="config")
    args = OmegaConf.create({})
    for key in confs.keys():
        args = OmegaConf.merge(args, confs[key])
    for key, value in args.items():
        if str(value).lower() == 'none':
            # 如果值是字符串 'none'（忽略大小写），则替换为 None
            args[key] = None
# Map "agent_n" to "policy_n" in

if __name__ == "__main__":
    args.num_agents = 2

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
    )

    run_rllib_example_script_experiment(base_config, args)
