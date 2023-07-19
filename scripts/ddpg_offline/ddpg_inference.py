import os
import random
import copy
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import rospy
import rospkg
from env_turtlebot import turtlebot_env

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action, min_action):
        super(Actor, self).__init__()
        self.max_action = torch.Tensor(max_action).to(device)
        self.min_action = torch.Tensor(min_action).to(device)
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        a = (self.max_action - self.min_action) / 2 * torch.tanh(self.l3(s)) + (self.max_action + self.min_action) / 2 # Scale the output to (min, max)
        # print("a: ", a)
        return a

class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.s1 = nn.Linear(state_dim, int(hidden_width/2))
        self.a1 = nn.Linear(action_dim, int(hidden_width/2))
        self.l1 = nn.Linear(hidden_width, hidden_width)
        self.l2 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        q_s = F.relu(self.s1(s))
        q_a = F.relu(self.a1(a))
        q = F.relu(self.l1(torch.cat([q_s, q_a], 1)))
        q = self.l2(q)
        return q

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, min_action):
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr1 = 3e-6  # learning rate
        self.lr2 = 6e-6  # learning rate
        self.noise_scale = 0.05  # Standard deviation of Gaussian noise
        self.noise_clip = 0.3

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action, min_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_width)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr1)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr2)

        self.MseLoss = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(device)
        # s = torch.where(torch.isinf(s), torch.tensor(5.), s)
        # print("s: ", s)
        a = self.actor(s)
        # Add noise to action for exploration
        noise = (torch.randn_like(a) * self.noise_scale).clamp(-self.noise_clip, self.noise_clip)
        a += noise
        a = torch.clamp(a, self.actor.min_action, self.actor.max_action)
        a = a.data.cpu().numpy().flatten()
        return a

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        batch_s = torch.FloatTensor(batch_s).to(device)
        batch_a = torch.FloatTensor(batch_a).to(device)
        batch_r = torch.FloatTensor(batch_r).unsqueeze(1).to(device)
        batch_s_ = torch.FloatTensor(batch_s_).to(device)
        batch_dw = torch.FloatTensor(batch_dw).unsqueeze(1).to(device)
        # print shape of batch_s, batch_a, batch_r, batch_s_, batch_dw
        # print('batch_s.shape: ', batch_s.shape)
        # print('batch_a.shape: ', batch_a.shape)
        # print('batch_r.shape: ', batch_r.shape)
        # print('batch_s_.shape: ', batch_s_.shape)
        # print('batch_dw.shape: ', batch_dw.shape)

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.MseLoss(target_Q, current_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        writer.add_scalar('Loss/critic_loss', critic_loss, i_episode)

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        writer.add_scalar('Loss/actor_loss', actor_loss, i_episode)
        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        print("update")


if __name__ == '__main__':
    env_name = "turtlebot3_ddpg"
    rospy.init_node('turtlebot3_ddpg', anonymous=True, log_level=rospy.WARN)
    env=turtlebot_env()
    rospy.loginfo("Gym environment done")

    # rospy.init_node('turtlebot3_world_ddpg', anonymous=True, log_level=rospy.WARN)
    # env=turtlebot_env()
    # rospy.loginfo("Gym environment done")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # obtain the path of the current folder
    current_folder_path = os.getcwd()
    outdir = current_folder_path + '/training_results/offline_ddpg_inference/'
    torch.manual_seed(0)

    run_num_pretrained = 1

    log_dir = outdir + "ddpg_logs" + "/log" + str(run_num_pretrained) 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    checkpoint_path = outdir + "saved_models/" + str(run_num_pretrained) 
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    writer = SummaryWriter(log_dir=log_dir)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_linear_velocity = float(env.action_space.high[0])
    max_angular_velocity = float(env.action_space.high[1])
    low_linear_velocity = float(env.action_space.low[0])
    low_angular_velocity = float(env.action_space.low[1])
    max_action = np.array([max_linear_velocity, max_angular_velocity])
    min_action = np.array([low_linear_velocity, low_angular_velocity])

    agent = DDPG(state_dim, action_dim, max_action, min_action)
    agent.actor = agent.actor.to(device)
    agent.critic = agent.critic.to(device)
    agent.actor_target = agent.actor_target.to(device)
    agent.critic_target = agent.critic_target.to(device)

    # Load the saved model
    agent.actor.load_state_dict(torch.load('/home/sezer/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/training_results/offline_ddpg_training/saved_models/check_1/DDPG_turtlebot3_continuous_59999_actor.pth'))
    agent.critic.load_state_dict(torch.load('/home/sezer/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/training_results/offline_ddpg_training/saved_models/check_1/DDPG_turtlebot3_continuous_59999_critic.pth'))

    # Set the model in evaluation mode
    agent.actor.eval()
    agent.critic.eval()

    num_episodes = 100000

    replay_buffer = ReplayBuffer(1e9)

    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("min_action={}".format(min_action))
    print("outdir={}".format(outdir))
    print("random action={}".format(env.action_space.sample()))

    for i_episode in range(num_episodes):
        state = env.reset()
        episode_steps = 0
        episode_reward = 0
        episode_arrive_reward = 0
        episode_distance_reward = 0
        episode_collision_reward = 0
        episode_velocity_reward = 0
        done = False
        for t in range(100):  # replace 100 by the length of an episode
            action = agent.choose_action(state)
            next_state, reward, done, other_reward = env.step(action)
            
            arrive_reward = other_reward["arrive_reward"]
            distance_reward = other_reward["distance_reward"]
            reward_collision = other_reward["reward_collision"]
            velocity_reward = other_reward["velocity_reward"]
            if not done and t == 99:
                arrive_reward = -2000
                reward += -2000
            episode_steps += 1
            episode_reward += reward
            episode_arrive_reward += arrive_reward
            episode_distance_reward += distance_reward
            episode_collision_reward += reward_collision
            episode_velocity_reward += velocity_reward
            # Store the transition in the replay buffer
            # replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state

            if done:
                break

        print("Episode: {}, episode_steps: {}, episode_reward: {}".format(i_episode, episode_steps, episode_reward))
        writer.add_scalar("episode_reward", episode_reward, i_episode)
        writer.add_scalar("arrive_reward", episode_arrive_reward, i_episode)
        writer.add_scalar("distance_reward", episode_distance_reward, i_episode)
        writer.add_scalar("reward_collision", episode_collision_reward, i_episode)
        writer.add_scalar("velocity_reward", episode_velocity_reward, i_episode)