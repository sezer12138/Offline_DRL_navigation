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
        self.lr1 = 3e-12  # learning rate
        self.lr2 = 6e-6  # learning rate
        self.noise_scale = 0.05  # Standard deviation of Gaussian noise

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action, min_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_width)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr1)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr2)

        self.MseLoss = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        # s = torch.where(torch.isinf(s), torch.tensor(5.), s)
        # print("s: ", s)
        a = self.actor(s).data.numpy().flatten()
        # Add noise to action for exploration
        noise = np.random.normal(0, self.noise_scale, size=a.shape)
        a += noise
        a = np.clip(a, self.actor.min_action, self.actor.max_action)
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
        writer.add_scalar('Q_values/current_Q_mean', current_Q.mean(), i_episode)
        writer.add_scalar('Q_values/current_Q_max', current_Q.max(), i_episode)


        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_output = self.actor(batch_s)
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        writer.add_scalar('Loss/actor_loss', actor_loss, i_episode)
        writer.add_scalar('Actor_output/actor_output_mean', actor_output.mean(), i_episode)
        writer.add_scalar('Actor_output/actor_output_max', actor_output.max(), i_episode)
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
    env_name = "turtlebot3_continuous"
    # rospy.init_node('turtlebot3_world_ddpg', anonymous=True, log_level=rospy.WARN)
    # env=turtlebot_env()
    # rospy.loginfo("Gym environment done")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # obtain the path of the current folder
    current_folder_path = os.getcwd()
    outdir = current_folder_path + '/training_results/offline_ddpg_training/'
    torch.manual_seed(0)

    state_dim = 40
    action_dim = 2
    max_action = np.array([0.5, 1.5])
    min_action = np.array([0.0, -1.5])
    num_episodes = 120000

    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("min_action={}".format(min_action))
    print("outdir={}".format(outdir))

    
    # run_num_pretrained = 2_2
    run_num_pretrained = "check_3"

    log_dir = outdir + "ddpg_logs" + "/log" + str(run_num_pretrained) 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    checkpoint_path = outdir + "saved_models/" + str(run_num_pretrained) 
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    agent = DDPG(state_dim, action_dim, max_action, min_action)
    agent.actor = agent.actor.to(device)
    agent.critic = agent.critic.to(device)
    agent.actor_target = agent.actor_target.to(device)
    agent.critic_target = agent.critic_target.to(device)

    replay_buffer = ReplayBuffer(1e9)

    # Load the tensors from file
    state_tensor, action_tensor, reward_tensor, new_state_tensor, done_tensor = torch.load('new_data1.pth')
    # Convert tensors to numpy arrays
    state_array = state_tensor.numpy()
    action_array = action_tensor.numpy()
    reward_array = reward_tensor.numpy()
    new_state_array = new_state_tensor.numpy()
    done_array = done_tensor.numpy()
    # Push the data into the replay buffer
    for i in range(len(state_array)):
        replay_buffer.push(state_array[i], action_array[i], reward_array[i], new_state_array[i], done_array[i])
    print("replay_buffer length={}".format(len(replay_buffer)))

    for i_episode in range(num_episodes):
        agent.learn(replay_buffer)

# Save the model
    torch.save(agent.actor.state_dict(), checkpoint_path + "/DDPG_{}_{}_actor.pth".format(env_name, i_episode))
    torch.save(agent.critic.state_dict(), checkpoint_path + "/DDPG_{}_{}_critic.pth".format(env_name, i_episode))