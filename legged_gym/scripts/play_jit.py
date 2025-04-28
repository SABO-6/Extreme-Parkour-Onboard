
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
from torch import nn
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer

from rsl_rl.modules import RecurrentDepthBackbone, DepthOnlyFCBackbone58x87

def load_model_path(device):

    path = '/home/zhanghb2023/project/extreme-parkour/legged_gym/logs/parkour_new/329-12-38/traced'
    base_model_name = '329-12-38-10000-base_jit.pt'
    vision_model_name = '329-12-38-10000-vision_weight.pt'

    base_model_path = os.path.join(path, base_model_name)
    vision_model_path = os.path.join(path, vision_model_name)

    base_model = torch.jit.load(base_model_path, map_location=device)

    vision_model = torch.load(vision_model_path, map_location=device)
    depth_backbone = DepthOnlyFCBackbone58x87(None, 32, 512)
    depth_encoder = RecurrentDepthBackbone(depth_backbone, None).to(device)
    depth_encoder.load_state_dict(vision_model['depth_encoder_state_dict'])

    print("Base model device:", next(base_model.parameters()).device)
    print("Depth encoder device:", next(depth_encoder.parameters()).device)

    return base_model, depth_encoder

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 5 if not args.save else 64 # 16
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.2,
                                    "parkour_hurdle": 0.2,
                                    "parkour_flat": 0., # 0
                                    "parkour_step": 0.2,
                                    "parkour_gap": 0.2, # 0.2 
                                    "demo": 0.2}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # print('observation shape: ', obs.shape)
    # print('observation: ', obs)

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)

    base_model, depth_encoder = load_model_path(device=env.device)
    base_model.eval()

    estimator = base_model.estimator.estimator
    hist_encoder = base_model.actor.history_encoder
    actor = base_model.actor.actor_backbone
    
    # else:
    #     policy = ppo_runner.get_inference_policy(device=env.device)
    # estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    # if env.cfg.depth.use_camera:
    #     depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    idx = 0
    obs_jit = torch.zeros((env.num_envs, 114), device=env.device)
    # n_proprio

    for i in range(10*int(env.max_episode_length)):
        if env.cfg.depth.use_camera:
            if infos["depth"] is not None:
                # obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                # obs_student[:, 6:8] = 0
                # print("depth shape: ", infos["depth"].shape)
                depth_latent_and_yaw = depth_encoder(infos["depth"], obs_jit[:, :env.cfg.env.n_proprio])
                depth_latent = depth_latent_and_yaw[:, :-2]
                depth_latent_len = depth_latent.shape[1]
                # n_depth_latent
                obs_jit[:, env.cfg.env.n_proprio:env.cfg.env.n_proprio+depth_latent_len] = depth_latent
                yaw = depth_latent_and_yaw[:, -2:]
            obs_jit[:, 6:8] = 1.5*yaw
        
        # 替换为估计的 lin_vel
        use_estimate_lin_vel = True
        if use_estimate_lin_vel:
            est_lin_vel = estimator(obs[:, :env.cfg.env.n_proprio])
            est_lin_vel_len = est_lin_vel.shape[1]
            # estimator_latent
            obs_jit[:, env.cfg.env.n_proprio+depth_latent_len:env.cfg.env.n_proprio+depth_latent_len+est_lin_vel_len] = est_lin_vel
        
        # history proprio latent 替换 priv_latent
        # history_latent = hist_encoder(obs[:, -env.cfg.env.history_len*env.cfg.env.n_proprio:])
        # print('history encoder code: ', hist_encoder.code)
        # print('history encoder graph: ', hist_encoder.graph)
        activation = nn.ELU()
        history_latent = hist_encoder(activation, obs[:, -env.cfg.env.history_len*env.cfg.env.n_proprio:].view(-1, env.cfg.env.history_len, env.cfg.env.n_proprio))
        history_latent_len = history_latent.shape[1]
        obs_jit[:, -history_latent_len:] = history_latent

        # actor 输入: n_proprio + n_depth_latent + estimator_latent + history_latent = 53 + 32 + 9 + 20 = 114
        # print('obs shape: ', obs_jit[env.lookat_id].shape)
        print('obs proprio: ', obs_jit[env.lookat_id, :env.cfg.env.n_proprio])
        actions = actor(obs_jit.detach())
        print('action: ', actions[env.lookat_id])
            
        # if i % 100 == 0:
        print('look at id: ', env.lookat_id)
        # print('observation shape: ', obs[env.lookat_id].shape) # num_env * 753 (proprio 53 + scandot 132 + priv_explicit（线速度） 9 + priv_latent （重力，摩擦系数） 29 + history latent 530)
        
        # num_observations = n_proprio + n_scan + n_priv + n_priv_latent + history_len*n_proprio

        # base angular velocity
        # 3 (base_ang_vel_x, base_ang_vel_y, base_ang_vel_z)
        # example tensor([ 0.3202, -0.1280,  0.2799], device='cuda:0', grad_fn=<SliceBackward0>)
        # print('base angular velocity: ', obs[env.lookat_id, :3])

        # base imu
        # 3 (roll, pitch, yaw)
        # example tensor([-0.0180,  0.1062,  0.0000], device='cuda:0', grad_fn=<SliceBackward0>)
        # print('base imu: ', obs[env.lookat_id, 3:6])

        # yaw error
        # 2 (delta_yaw, next_delta_yaw)
        # print('estimate yaw error: ', obs[env.lookat_id, 6:8])

        # commands
        # 3 (cmd_vx, cmd_vy, cmd_omega) 只需要最后一个元素是 前进速度 命令 
        # print('commands: ', obs[env.lookat_id, 8:11])

        # env class 17？
        # if i % 100 == 0:
        #     print('env class == 17?', obs[env.lookat_id, 11:13])

        # contact filt
        # example tensor([ 0.5000, -0.5000, -0.5000,  0.5000], device='cuda:0', grad_fn=<SliceBackward0>)
        # 猜测接触是 0.5, 没接触是 -0.5
        # print('contact filt: ', obs[env.lookat_id, -4:]) # 4


        # 132 scandot 
        # print('scandot latent: ', ppo_runner.get_depth_actor_scandots_latent(obs[env.lookat_id, :].unsqueeze(0), device=env.device)) # 32
        # print('depth latent (estimate of scandot latent): ', depth_latent[env.lookat_id]) # 32

        # 9 (lin_vel_x, lin_vel_y, lin_vel_z, vx, vy, vz, roll, pitch, yaw)
        # example (1.1, 2.2, 3.3, 0, 0, 0, 0, 0, 0)
        # print('lin velocity: ', obs[env.lookat_id, env.cfg.env.n_proprio+env.cfg.env.n_scan:env.cfg.env.n_proprio+env.cfg.env.n_scan+env.cfg.env.n_priv].cpu().numpy()) 
        # print('estimate lin velocity: ', ppo_runner.get_estimator_inference_policy()(obs[env.lookat_id, :env.cfg.env.n_proprio].unsqueeze(0)).cpu().numpy()) # 估计的线速度

        # 29 (mass gravity[4], friction[1], motor_strength[12], motor_strength[12])
        # 都是常量，mass gravity 全为 0，后面的 motor_strength 值不相同
        # example: [ 0.          0.          0.          0.          0.7687572  -0.07551169
                    # -0.1896506  -0.14859194  0.16632974  0.00752091  0.0407455  -0.03080791
                    # 0.06396985 -0.14890838 -0.1096378  -0.16341943 -0.0230974   0.11250949
                    # 0.0413276  -0.09061724 -0.00917965  0.1176132  -0.00810266 -0.05108631
                    # -0.17121857 -0.08175284 -0.16733825  0.137851    0.13111997]
        # print('private latent: ', obs[env.lookat_id, env.cfg.env.n_proprio+env.cfg.env.n_scan+env.cfg.env.n_priv:env.cfg.env.n_proprio+env.cfg.env.n_scan+env.cfg.env.n_priv+env.cfg.env.n_priv_latent].cpu().numpy())
        # print('latent of private latent: ', ppo_runner.get_depth_actor_priv_latent(obs[env.lookat_id].unsqueeze(0), device=env.device))
        # print('latent of history proprio (estimate of private latent): ', ppo_runner.get_depth_actor_hist_latent(obs[env.lookat_id].unsqueeze(0), device=env.device))


        obs, _, rews, dones, infos = env.step(actions.detach())
        obs_jit[:, :env.cfg.env.n_proprio] = obs[:, :env.cfg.env.n_proprio]

        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        # print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
        #       "cmd vx", env.commands[env.lookat_id, 0].item(),
        #       "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        
        id = env.lookat_id

# output_dir = '/home/zhanghb2023/project/extreme-parkour/legged_gym/logs/depth_images'
# os.makedirs(output_dir, exist_ok=True)
# def turn_image(obs, idx):
#     obs = obs.cpu().numpy()
#     print('obs shape: ', obs.shape)
#     print('image example: ', obs[0, :])
#     # obs = obs.reshape(58, 87) + 0.5
#     # obs = cv2.rotate(obs, cv2.ROTATE_90_CLOCKWISE)
#     # obs = cv2.flip(obs, 1)
#     cv2.imwrite(os.path.join(output_dir, f'obs_image_{idx}.png'), obs * 255)
#     # cv2.imwrite("depth_images/depth_{}.png".format(idx), obs)
#     print(f'successfully save image in {output_dir}.')
        

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
