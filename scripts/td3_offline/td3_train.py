import copy
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, min_action):
        super(Actor, self).__init__()

        self.max_action = torch.Tensor(max_action).to(device)
        self.min_action = torch.Tensor(min_action).to(device)
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = (self.max_action - self.min_action) / 2 * torch.tanh(self.l3(a)) + (self.max_action + self.min_action) / 2 # Scale the output to (min, max)
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.s1 = nn.Linear(state_dim, 128)
        self.a1 = nn.Linear(action_dim, 128)
        self.s2 = nn.Linear(state_dim, 128)
        self.a2 = nn.Linear(action_dim, 128)

        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 1)

        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 1)

    def forward(self, state, action):
        q1_s = F.relu(self.s1(state))
        q1_a = F.relu(self.a1(action))
        q1_sa = torch.cat([q1_s, q1_a], 1)
        q1 = F.relu(self.l1(q1_sa))
        q1 = self.l2(q1)

        q2_s = F.relu(self.s2(state))
        q2_a = F.relu(self.a2(action))
        q2_sa = torch.cat([q2_s, q2_a], 1)
        q2 = F.relu(self.l3(q2_sa))
        q2 = self.l4(q2)

        return q1, q2

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

class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        min_action,
    ):
        self.actor = Actor(state_dim, action_dim, max_action, min_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.1
        self.noise_clip = 0.5
        self.policy_freq = 5

        self.total_it = 0

    def select_action(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
        a = self.actor(state)
        # Add noise to action for exploration
        noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        a += noise
        a = torch.clamp(a, self.actor.min_action, self.actor.max_action)
        # convert to numpy
        a = a.data.numpy().flatten()
        return a

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        batch_s, batch_a, batch_r, batch_s_, batch_dw = replay_buffer.sample(batch_size)  # Sample a batch

        batch_s = torch.FloatTensor(batch_s).to(device)
        batch_a = torch.FloatTensor(batch_a).to(device)
        batch_r = torch.FloatTensor(batch_r).unsqueeze(1).to(device)
        batch_s_ = torch.FloatTensor(batch_s_).to(device)
        batch_dw = torch.FloatTensor(batch_dw).unsqueeze(1).to(device)

        with torch.no_grad():
            noise = (torch.randn_like(batch_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(batch_s_) + noise).clamp(self.actor.min_action, self.actor.max_action)

            target_Q1, target_Q2 = self.critic_target(batch_s_, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = batch_r + (1 - batch_dw) * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(batch_s, batch_a)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        writer.add_scalar('Loss/critic_loss', critic_loss, i_episode)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            q1, q2 = self.critic(batch_s, self.actor(batch_s))
            q = torch.min(q1, q2)
            actor_loss = -q.mean()
            writer.add_scalar('Loss/actor_loss', actor_loss, i_episode)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

if __name__ == '__main__':
    env_name = "turtlebot3_continuous"
    current_folder_path = os.getcwd()
    outdir = current_folder_path + '/training_results/offline_TD3/'
    torch.manual_seed(0)

    state_dim = 40
    action_dim = 2
    max_action = np.array([0.5, 1.5])
    min_action = np.array([0.0, -1.5])
    num_episodes = 60000

    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("min_action={}".format(min_action))
    print("outdir={}".format(outdir))

    run_num_pretrained = 0
    log_dir = outdir + "td3_logs" + "/log" + str(run_num_pretrained) 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    checkpoint_path = outdir + "saved_models/" + str(run_num_pretrained) 
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    writer = SummaryWriter(log_dir=log_dir)

    agent = TD3(state_dim, action_dim, max_action, min_action)
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
        agent.train(replay_buffer)

    # Save the model
    torch.save(agent.actor.state_dict(), checkpoint_path + "/TD3_{}_{}_actor.pth".format(env_name, i_episode))
    torch.save(agent.critic.state_dict(), checkpoint_path + "/TD3_{}_{}_critic.pth".format(env_name, i_episode))
    