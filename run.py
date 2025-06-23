from pathlib import Path
from pprint import pprint

import numpy as np
from ray import tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME
from shared_memory_dict import SharedMemoryDict
from swankit.env import is_windows

from config import ConfigSingleton
import os
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

from core.custom_flatten_observations import CustomFlattenObservations
from core.custom_ppo_rl_module import CustomDefaultPPOTorchRLModule
from envs.env_4 import Env_4
from envs.env_5 import Env_5
from envs.env_6 import Env_6
from utils.csv_util import append_data, write_data

torch, _ = try_import_torch()
from ray.rllib.connectors.env_to_module.flatten_observations import FlattenObservations
from ray.tune.registry import get_trainable_cls, register_env  # noqa
from run_helper import train
from utils.evaluate import evaluate_v2
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)
'''
Gaming the Quantum bit Placement with AI
'''

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
    if args.enable_cnn:
        conv_filters = [
            [32, 3, 1],  # 过滤器数量，卷积核大小 步幅
            [64, 3, 2],  # 过滤器数量，卷积核大小 步幅
            [128, 3, 2],  # 过滤器数量，卷积核大小 步幅
            [256, 3, 2],  # 过滤器数量，卷积核大小 步幅
        ]
    else:
        conv_filters = None

    model_config = DefaultModelConfig(
        # if use lstm, the AddTimeDimToBatchAndZeroPad connector will throw error
        use_lstm=args.use_lstm
        ,conv_filters=conv_filters
        ,conv_activation=args.conv_activation
        ,fcnet_hiddens=args.fcnet_hiddens
        ,head_fcnet_hiddens = args.head_fcnet_hiddens
        ,fcnet_activation=args.fcnet_activation
       # ,fcnet_kernel_initializer = 'kaiming_uniform_'
    )
    rl_module_specs = {
            'policy_{}'.format(i): RLModuleSpec(
                model_config=model_config,
                module_class = CustomDefaultPPOTorchRLModule
            ) for i in range(1, int(args.num_qubits) + 1)
    }
    return rl_module_specs

def get_policys():
    policies = {'policy_{}'.format(i) for i in range(1, int(args.num_qubits) + 1)}
    return policies

def save_state():
    smd = SharedMemoryDict(name='env', size=1024)
    for k in smd.keys():
        print(f'key = {k}, value = {smd[k]}')
    if 'best_state'in smd.keys():
        state = smd['best_state']
        dist = smd['min_dist']
        path = Path(args.results_evaluate_path, (args.time_id + '_good_results.csv'))
        write_data(file_path=path, data=[[dist]])
        state = np.array(state).astype(int)
        append_data(file_path=path, data=state)
    else:
        print('no best state found in shared memory dict')
if __name__ == "__main__":

    #set custom run config before init args
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--iter", '-i',type=int, help="train iter", default=None)
    parser.add_argument("--swanlab", '-w',type=bool, help="enable_swanlab",default=True)
    parser.add_argument("--wandb_group", '-wg',type=str, help="wandb_group",default='default')
    parser.add_argument("--checkpoint", '-c',type=str, help="best checkpoint",default=None)
    parser.add_argument("--run_name", '-name',type=str, help="wandb project run name",default=None)

    cmd_args = parser.parse_args()
    args = ConfigSingleton().get_args()

    if is_windows():
        print('run on windows')
        cmd_args.swanlab = False
        args.enable_cnn = False
    stop = {
            # f"{ENV_RUNNER_RESULTS}/{NUM_ENV_STEPS_SAMPLED_LIFETIME}": (
            #     args.stop_timesteps
            # ),
            TRAINING_ITERATION: args.stop_iters,
    }

    if cmd_args.iter is not None:
        stop[TRAINING_ITERATION] = cmd_args.iter

    Envs = {
        "Env5": Env_5,
        "Env6": Env_6,
        # 添加更多环境映射...
    }
    base_config = (
        # get_trainable_cls(args.algo_class)
        # .get_default_config()
        PPOConfig()
        .environment(
            env=Envs[f'Env{args.env_version}'],
            env_config={"key": "value"},
        )
        .training(
            use_gae = True,
            lr=tune.grid_search(args.lr_grid),
            # lr_schedule=tune.grid_search([
            #     [[0, 5.0e-5], [4000 * 100, 5.0e-5], [4000 * 200, 1.0e-5]]
            # ),
            train_batch_size = 1024*16,
            minibatch_size=512*16,
            gamma=tune.grid_search(args.gamma_grid),
            lambda_ = args.lambda_,
            entropy_coeff =args.entropy_coeff,
            vf_loss_coeff = args.vf_loss_coeff,
            kl_target =args.kl_target,
        )
        .multi_agent(
            # Define two policies.
            policies=get_policys(),
            # Map agent "player1" to policy "player1" and agent "player2" to policy
            # "player2".
            policy_mapping_fn=policy_mapping_fn,

        ).rl_module(
            rl_module_spec=MultiRLModuleSpec(
               # multi_rl_module_class=ActionMaskingTorchRLModule,
                rl_module_specs=get_rl_module_specs()
            ),
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
            env_to_module_connector=lambda env: CustomFlattenObservations(multi_agent=True),
            num_gpus_per_env_runner = 0
            #num_env_runners=args.num_env_runners,
        )
        .learners(
            num_learners=args.num_learners,
            num_gpus_per_learner= args.num_gpus_per_learner
        )
    )
    # pprint(base_config.to_dict())
    # .rl_module(
    #     rl_module_spec=MultiRLModuleSpec(rl_module_specs={
    #         "learning_policy": RLModuleSpec(),
    #         "random_policy": RLModuleSpec(rl_module_class=RandomRLModule),
    #     }),
    # )
    #

    #
    if cmd_args.checkpoint is not None:
        results = cmd_args.checkpoint
    else:
        results = train(config = base_config, cmd_args = cmd_args,args=args,enable_swanlab=cmd_args.swanlab,stop=stop)
        save_state()
    evaluate_v2(base_config,args,results)

    # print(base_config.to_dict())
    #
    # results = r'C:\Users\ADMINI~1\AppData\Local\Temp\checkpoint_tmp_71b632214bf64e1f818c612bcaadcb3b'
    # evaluate_v2(base_config,args,results)
