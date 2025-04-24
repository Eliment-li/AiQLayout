from pettingzoo.butterfly import pistonball_v6

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

register_env(
    "pistonball",
    lambda cfg: PettingZooEnv(pistonball_v6.env(num_floors=cfg.get("n_pistons", 20))),
)

config = (
    PPOConfig()
    .environment("pistonball", env_config={"n_pistons": 30})
)