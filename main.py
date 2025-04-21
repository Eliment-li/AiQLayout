import time

import hydra
import ray
from gymnasium import register
from omegaconf import DictConfig, OmegaConf
import os

from ray import tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME, EPISODE_RETURN_MEAN, ENV_RUNNER_RESULTS
from ray.tune import CLIReporter
from hydra import initialize, compose
from omegaconf import DictConfig
args = None
# 手动初始化 Hydra
with initialize(config_path="conf", job_name="config"):
    # 加载配置文件
    args = compose(config_name="config")
# 设置环境变量
os.environ["HYDRA_FULL_ERROR"] = "1"

# 定义一个普通函数来实现策略映射
def policy_mapping_fn(aid, episode):
    # 提取 player 的编号，并映射到对应的策略
    player_number = int(aid.split('_')[1])  # 从 agent ID 提取数字部分
    return f"p{player_number}"  # 返回对应的策略名称

def init_algo_config(args):
    args = args.run
    # 最简单的环境
    register(
        id='env_1',
        # entry_point='core.envs.circuit_env:CircuitEnv',
        entry_point='env.custom_environment:CustomEnvironment',
        max_episode_steps=2000000,
    )

    config = (
        PPOConfig()
        .environment(env="env_1")
        # .env_runners(
        #     env_to_module_connector=lambda env: FlattenObservations(multi_agent=True),
        # )
        .multi_agent(
            policy_mapping_fn=policy_mapping_fn,
            policies={"p0", "p1"}
        )
        .training(
            vf_loss_coeff=0.005,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "p0": RLModuleSpec(),
                    "p1": RLModuleSpec(),
                }
            ),
            model_config=DefaultModelConfig(
                use_lstm=args.use_lstm,
                # Use a simpler FCNet when we also have an LSTM.
                fcnet_hiddens=[32] if args.use_lstm else [256, 256],
                lstm_cell_size=256,
                max_seq_len=15,
                vf_share_layers=True,
            ),
        )
    )

    return config

def train(config, args: DictConfig):
    args =args.run
    # Auto-configure a CLIReporter (to log the results to the console).
    # Use better ProgressReporter for multi-agent cases: List individual policy rewards.
    progress_reporter = CLIReporter(
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

    # Run the actual experiment (using Tune).
    try:
        start_time = time.time()
        results = tune.Tuner(
           config.algo_class,
            param_space=config,
            run_config=tune.RunConfig(
                stop={
             TRAINING_ITERATION: args.stop_iters,
                 #NUM_ENV_STEPS_SAMPLED_LIFETIME: 1000,
                },
                verbose=args.verbose,
                #callbacks=tune_callbacks,
                # checkpoint_config=tune.CheckpointConfig(
                #     checkpoint_frequency=args.checkpoint_freq,
                #     checkpoint_at_end=args.checkpoint_at_end,
                # ),
                progress_reporter=progress_reporter,
            ),
            # tune_config=tune.TuneConfig(
            #     num_samples=args.num_samples,
            #     max_concurrent_trials=args.max_concurrent_trials,
            #     scheduler=scheduler,
            # ),
        ).fit()
        time_taken = time.time() - start_time
    except Exception as e:
        print(f"Error occurred: {e}")
        time_taken = None
    finally:
        # Perform any necessary cleanup or finalization here
        print(f"Total time taken: {time_taken} seconds")
        ray.shutdown()


if __name__ == "__main__":
    config = init_algo_config(args)
    train(config=config,args = args)

