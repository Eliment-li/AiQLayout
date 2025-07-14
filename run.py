import re
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import ray
import redis
from ray import tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from shared_memory_dict import SharedMemoryDict
from swankit.env import is_windows
import argparse

from ray.rllib.utils.framework import try_import_torch
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

import config
from core.custom_flatten_observations import CustomFlattenObservations
from core.custom_ppo_rl_module import CustomDefaultPPOTorchRLModule
from envs.env_5 import Env_5
from envs.env_6 import Env_6
from utils.file.csv_util import append_data, write_data
from utils.file.file_util import get_root_dir

torch, _ = try_import_torch()
from ray.tune.registry import get_trainable_cls, register_env  # noqa
from run_helper import train
from utils.evaluate import evaluate_v2
import os
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
def get_rl_module_specs(args):
    if args.enable_cnn:
        # conv_filters = [
        #     [32, 3, 1],  # 过滤器数量，卷积核大小 步幅
        #     [64, 3, 2],  # 过滤器数量，卷积核大小 步幅
        #     [128, 3, 2],  # 过滤器数量，卷积核大小 步幅
        #     #[256, 3, 2],  # 过滤器数量，卷积核大小 步幅
        # ]
        conv_filters = args.conv_filters
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
    smd = SharedMemoryDict(name='env', size=10240)
    for k in smd.keys():
        print(f'key = {k}, value = {smd[k]}')
    if 'best_state'in smd.keys():
        state = smd['best_state']
        min_dist = smd['min_dist']
        min_dist = f'min_dist = {min_dist}'
        init_dist = smd['init_dist']
        init_dist = f'init_dist = {init_dist}'
        path = Path(args.results_evaluate_path, (args.time_id + '_good_results.csv'))
        write_data(file_path=path, data=[[min_dist]])
        append_data(file_path=path, data=[[init_dist]])
        state = np.array(state).astype(int)
        state = repr(state)
        append_data(file_path=path, data=[[state]])
    else:
        print('no best state found in shared memory dict')


def run(args,cmd_args):
    Envs = {
        "Env5": Env_5,
        "Env6": Env_6,
        # 添加更多环境映射...
    }
    # set custom run config before init args

    # enable redis on linux
    # if not is_windows() and args.enable_redis:
    #     os.environ['RAY_REDIS_ADDRESS'] = '127.0.0.1:6379'
    #     flush_redis()

    stop = {
        TRAINING_ITERATION: args.stop_iters,
    }

    if cmd_args.iter is not None:
        stop[TRAINING_ITERATION] = cmd_args.iter

    base_config = (
        # get_trainable_cls(args.algo_class)
        # .get_default_config()
        PPOConfig()
        .environment(
            env=Envs[f'Env{args.env_version}'],
            env_config={
              'num_qubits':  args.num_qubits,
              'chip_rows':  args.chip_rows,
              'chip_cols':  args.chip_cols,
              'lsi_file_path':  args.lsi_file_path,
              'env_max_step':  args.env_max_step,
              'layout_type':  args.layout_type,
              'rf_version':  args.rf_version,
              'gamma':  args.gamma,
              'reward_scaling':  args.reward_scaling,
            },
        )
        .training(
            use_gae=True,
            lr=tune.grid_search(args.lr_grid),
            # lr_schedule=tune.grid_search([
            #     [[0, 5.0e-5], [4000 * 100, 5.0e-5], [4000 * 200, 1.0e-5]]
            # ),
            train_batch_size=1024 * 16,
            minibatch_size=512 * 16,
            gamma=tune.grid_search(args.gamma_grid),
            lambda_=args.lambda_,
            entropy_coeff=args.entropy_coeff,
            vf_loss_coeff=args.vf_loss_coeff,
            kl_target=args.kl_target,
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
                rl_module_specs=get_rl_module_specs(args)
            ),
            # algorithm_config_overrides_per_module={
            #     "policy_1": PPOConfig.overrides(
            #         gamma=0.85
            #     ),
            #     "policy_2": PPOConfig.overrides(
            #         lr=0.00001
            #     ),
            # },
            # model_config=get_model_config()
        ).env_runners(
            env_to_module_connector=lambda env: CustomFlattenObservations(multi_agent=True),
            num_gpus_per_env_runner=0
            # num_env_runners=args.num_env_runners,
        )
        .learners(
            num_learners=args.num_learners,
            num_gpus_per_learner=args.num_gpus_per_learner
        )
    )
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
        results = train(config=base_config, cmd_args=cmd_args, args=args, enable_swanlab=cmd_args.swanlab, stop=stop)
        save_state()
    evaluate_v2(base_config, args, results)
    #evaluate_v2(base_config,args,r'C:\Users\ADMINI~1\AppData\Local\Temp\checkpoint_tmp_20e0fb06eee944b1a3075b26cb2e39f5')


def get_dynamic_conf(lsi_file_path,num_qubits ):
    # 0.8 * logical_cpu_cores
    # cfg['num_cpus'] = int(os.cpu_count() * 0.7)
    cfg={}
    # default number_learner is 1, thus  set num_gpus_per_learner to 1 is fine
    cfg['num_gpus_per_learner'] = 1 if torch.cuda.is_available() else 0
    cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    cfg['time_id'] = datetime.now().strftime('%m%d_%H%M')
    cfg['output'] = None
    #cfg['tensorboard_path'] = p / 'results' / 'tensorboard'
    path = Path(get_root_dir()) / 'results' / 'evaluate' / cfg['time_id']
    cfg['results_evaluate_path'] = str(path)
    os.makedirs(path, exist_ok=True)
    match = re.search(r"LSI.*", lsi_file_path)
    circuit = match.group()
    cfg['wandb_run_name'] = cfg['time_id'] + '_' + circuit
    cfg['num_qubits'] = num_qubits
    cfg['enable_cnn'] = not is_windows()

    return cfg

def init_args(exp):
    args = config.RedisConfig()
    args.wait_until_initialized()
    args.update_redis(exp)
    args.update_redis(get_dynamic_conf(exp['lsi_file_path'], exp['num_qubits']))
    return args
if __name__ == "__main__":

    path = Path(get_root_dir()) / 'conf'
    redis_config = config.RedisConfig()
    redis_config.flush()  # 清空 Redis 数据库
    # 初始化（只做一次）
    redis_config.initialize(path)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--iter", '-i', type=int, help="train iter", default=None)
    parser.add_argument("--swanlab", '-w', type=bool, help="enable_swanlab", default=True)
    parser.add_argument("--wandb_group", '-wg', type=str, help="wandb_group", default='default')
    parser.add_argument("--checkpoint", '-c', type=str, help="best checkpoint", default=None)
    parser.add_argument("--run_name", '-name', type=str, help="wandb project run name", default=None)
    cmd_args = parser.parse_args()
    # if is_windows():
    #     print('run on windows')
    #     cmd_args.swanlab = False
    for i in [4,6,8]:
        time.sleep(5)  # Wait for a few seconds to ensure all processes are cleaned up
        #ray.init(local_mode=False)
        SharedMemoryDict(name='ConfigSingleton', size=10240).cleanup()
        SharedMemoryDict(name='env', size=10240).cleanup()
        try:
            exp = {
                'lsi_file_path':f'assets/circuits/qft/LSI_qftentangled_indep_qiskit_{i}.lsi',
                'num_qubits': i,
            }
            print(f"Running experiment with {i} qubits...")
            args =  init_args(exp)
            run(args, cmd_args)
            #ray.shutdown()
            print(f"Finsh experiment with {i} qubits...")
        except Exception as e:
            traceback.print_exc()
        finally:
            # smd = SharedMemoryDict(name='env', size=10240)
            # smd.cleanup()
            # smd.shm.unlink()
            ray.shutdown()







