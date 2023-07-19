import copy
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
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

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
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

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

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            q1, q2 = self.critic(batch_s, self.actor(batch_s))
            q = torch.min(q1, q2)
            actor_loss = -q.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
