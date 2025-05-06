from ray import tune
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module import MultiRLModule, MultiRLModuleSpec, RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from config import ConfigSingleton
import os
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from utils.evaluate import evaluate

torch, _ = try_import_torch()
from ray.rllib.connectors.env_to_module.flatten_observations import FlattenObservations
from ray.tune.registry import get_trainable_cls, register_env  # noqa
from envs.env_0 import Env_0
from envs.env_1 import Env_1
from run_helper import train

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
    model_config = DefaultModelConfig(
        use_lstm=False,
        #conv_filters= [[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]]
        # conv_activation='relu',
        # fcnet_hiddens=[256,256],
        # fcnet_activation='relu',

    )
    return model_config

if __name__ == "__main__":

    args = ConfigSingleton().get_args()
    #policy_1, policy_2 ... policy_args.num_agents
    policies = {'policy_{}'.format(i+1) for i in range(int(args.num_qubits))}

    base_config = (
        get_trainable_cls(args.algo_class)
        .get_default_config()
        .environment(
            Env_0,
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
            policies=policies,
            # Map agent "player1" to policy "player1" and agent "player2" to policy
            # "player2".
            policy_mapping_fn=policy_mapping_fn,

        ).rl_module(
            rl_module_spec=MultiRLModuleSpec(rl_module_specs={
                "policy_1": RLModuleSpec(model_config = get_model_config()),
                "policy_2": RLModuleSpec(model_config = get_model_config()),
            }),
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
        )


    )
    # .rl_module(
    #     rl_module_spec=MultiRLModuleSpec(rl_module_specs={
    #         "learning_policy": RLModuleSpec(),
    #         "random_policy": RLModuleSpec(rl_module_class=RandomRLModule),
    #     }),
    # )

    results = train(base_config, args)
    evaluate(base_config,args,results)
    #evaluate(base_config,args,r'C:\Users\ADMINI~1\AppData\Local\Temp\checkpoint_tmp_2a7882f0dd954ac381bbfc4f890c08d7')