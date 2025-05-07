from ray import tune
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module import MultiRLModule, MultiRLModuleSpec, RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from config import ConfigSingleton
import os
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from utils.evaluate import evaluate, evaluate_v2

torch, _ = try_import_torch()
from ray.rllib.connectors.env_to_module.flatten_observations import FlattenObservations
from ray.tune.registry import get_trainable_cls, register_env  # noqa
from envs.env_0 import Env_0
from envs.env_1 import Env_1
from run_helper import train

args = ConfigSingleton().get_args()
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
def get_model_config():
    ConvFilterSpec = [
        [16, 3, 1],  # 第一层：32 个过滤器，卷积核大小为 3x3，步幅为 1x1
        [32, 3, 2],  # 第二层：64 个过滤器，卷积核大小为 3x3，步幅为 2x2
        [64, 5, 1]  # 第三层：128 个过滤器，卷积核大小为 5x5，步幅为 1x1
    ]

    model_config = DefaultModelConfig(
        # if use lstm, the AddTimeDimToBatchAndZeroPad connector will throw error
        use_lstm=False,
        conv_filters= ConvFilterSpec
        # conv_activation='relu',
        # fcnet_hiddens=[256,256],
        # fcnet_activation='relu',

    )
    return model_config


    # specific the rl module
def get_rl_module_specs():
    rl_module_specs = {
            'policy_{}'.format(i): RLModuleSpec(model_config=get_model_config()) for i in
            range(1, int(args.num_qubits) + 1)
    }
    return rl_module_specs

def get_policys():
    policies = {'policy_{}'.format(i) for i in range(1, int(args.num_qubits) + 1)}
    return policies

if __name__ == "__main__":
    base_config = (
        get_trainable_cls(args.algo_class)
        .get_default_config()
        .environment(
            env=Env_1,
            env_config={"key": "value"},
        )
        .training(
            lr=tune.grid_search(args.lr_grid),
            gamma=tune.grid_search(args.gamma_grid),
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

    results = train(base_config, args)
    evaluate_v2(base_config,args,results)

    # results = r'C:\Users\90471\AppData\Local\Temp\checkpoint_tmp_fb68c2853b8643d88b886094a6a1d32c'
    # evaluate_v2(base_config,args,r'C:\Users\90471\AppData\Local\Temp\checkpoint_tmp_fb68c2853b8643d88b886094a6a1d32c')