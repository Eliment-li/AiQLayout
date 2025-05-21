from ray import tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module import MultiRLModule, MultiRLModuleSpec, RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME

from config import ConfigSingleton
import os
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

from envs.env_2 import Env_2

torch, _ = try_import_torch()
from ray.rllib.connectors.env_to_module.flatten_observations import FlattenObservations
from ray.tune.registry import get_trainable_cls, register_env  # noqa
from envs.env_1 import Env_1
from run_helper import train
from utils.evaluate import evaluate, evaluate_v2

'''
Gaming the Quantum bit Placement with AI
'''

# def new_env():
#     return  Env_1()
# register_env("Env_0", new_env)
def policy_mapping_fn(agent_id, episode, **kwargs):
    try:
        agent_number = agent_id.split('_')[1]  # Split and get the number part
        return f"policy_{agent_number}"       # Return the mapped policy name
    except IndexError:
        raise ValueError(f"Invalid agent_id format: {agent_id}. Expected 'agent_n'.")

'''
DefaultModelConfig:
MLP stacks
Head configs (e.g. policy- or value function heads)
Conv2D stacks
Continuous action settings
LSTM settings
'''
    # specific the rl module
def get_rl_module_specs():
    ConvFilterSpec = [
        [16, 2, 1,'same'],  # 过滤器数量，卷积核大小 步幅
        [32, 3, 1],  # 过滤器数量，卷积核大小 步幅
        [64, 3, 1],  # 过滤器数量，卷积核大小 步幅
    ]

    model_config = DefaultModelConfig(
        # if use lstm, the AddTimeDimToBatchAndZeroPad connector will throw error
        use_lstm=False
        ,conv_filters=ConvFilterSpec
        ,conv_activation='relu'
        ,fcnet_hiddens=[1024, 1024, 512]
        ,head_fcnet_hiddens = [512,256]
        ,fcnet_activation='relu'
    )
    rl_module_specs = {
            'policy_{}'.format(i): RLModuleSpec(model_config=model_config) for i in
            range(1, int(args.num_qubits) + 1)
    }
    return rl_module_specs

def get_policys():
    policies = {'policy_{}'.format(i) for i in range(1, int(args.num_qubits) + 1)}
    return policies

if __name__ == "__main__":
    #set custom run config before init args
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--iter", '-i',type=int, help="train iter", default=None)
    parser.add_argument("--wandb", '-w',type=bool, help="enable_wandb",default=False)
    parser.add_argument("--checkpoint", '-c',type=str, help="best checkpoint",default=None)

    cmd_args = parser.parse_args()

    args = ConfigSingleton().get_args()
    stop = {
            f"{ENV_RUNNER_RESULTS}/{NUM_ENV_STEPS_SAMPLED_LIFETIME}": (
                args.stop_timesteps
            ),
            TRAINING_ITERATION: args.stop_iters,
    }

    if cmd_args.iter is not None:
        stop[TRAINING_ITERATION] = cmd_args.iter

    print(stop)
    base_config = (
        get_trainable_cls(args.algo_class)
        .get_default_config()
        .environment(
            env=Env_2,
            env_config={"key": "value"},
        )
        .training(
            lr=tune.grid_search(args.lr_grid),
            gamma=tune.grid_search(args.gamma_grid),
            lambda_ = 0.9,
            entropy_coeff = 0.01,
            vf_loss_coeff = 0.5
            # model={
            #     # "fcnet_hiddens":args.fcnet_hiddens ,
            #     "fcnet_hiddens": tune.grid_search(args.fcnet_hiddens_grid),  # args.fcnet_hiddens,
            #     "fcnet_activation": tune.grid_search(args.fcnet_activation),
            #     "use_attention": False,
            # },

            # step = iteration * 4000
            #     lr_schedule= tune.grid_search([
            #     [[0, 5.0e-5], [4000*100, 5.0e-5],[4000*200,1.0e-5]],
            #    # [[0, 0.001], [1e9, 0.0005]],
            # ]),
        )
        .multi_agent(
            # Define two policies.
            policies=get_policys(),
            # Map agent "player1" to policy "player1" and agent "player2" to policy
            # "player2".
            policy_mapping_fn=policy_mapping_fn,

        ).rl_module(
            rl_module_spec=MultiRLModuleSpec(rl_module_specs=get_rl_module_specs()),
            # algorithm_config_overrides_per_module={
            #     "policy_1": PPOConfig.overrides(
            #         gamma=0.85
            #     ),
            #     "policy_2": PPOConfig.overrides(
            #         lr=0.00001
            #     ),
            # },
            #model_config=get_model_config()
        ).env_runners(
            env_to_module_connector=lambda env: FlattenObservations(multi_agent=True),
            num_gpus_per_env_runner = 0
            #num_env_runners=args.num_env_runners,
        ).learners(
            num_learners=args.num_learners,
            num_gpus_per_learner= args.num_gpus_per_learner
        )
    )
    # .rl_module(
    #     rl_module_spec=MultiRLModuleSpec(rl_module_specs={
    #         "learning_policy": RLModuleSpec(),
    #         "random_policy": RLModuleSpec(rl_module_class=RandomRLModule),
    #     }),
    # )
    #
    if cmd_args.checkpoint is not None:
        results = cmd_args.checkpoint
    else:
        results = train(config = base_config, args=args,enable_wandb=cmd_args.wandb,stop=stop)
    evaluate_v2(base_config,args,results)
    #
    # results = r'C:\Users\ADMINI~1\AppData\Local\Temp\checkpoint_tmp_1afe49475c1349148e81b6b30ee9ae6d'
    # evaluate_v2(base_config,args,results)
