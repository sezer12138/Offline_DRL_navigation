import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import rospy
import random
from torch.utils.tensorboard import SummaryWriter
from env_turtlebot import turtlebot_env
import os 

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, min_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, action_dim)

        self.max_action = max_action
        self.min_action = min_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = torch.tanh(self.l3(a))
        log_std = torch.zeros_like(mean)
        std = torch.exp(log_std)

        # Create a Normal distribution and apply the squashing function
        normal_dist = torch.distributions.Normal(mean, std)
        x_t = normal_dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * (self.max_action - self.min_action) / 2 + (self.max_action + self.min_action) / 2

        # Compute the log probability
        log_prob = normal_dist.log_prob(x_t)
        
        # Apply the log derivative trick for the change of variable
        log_prob -= torch.log((self.max_action - self.min_action) * (1 - y_t.pow(2)) + 1e-6)
        # print("log_prob={}".format(log_prob))
        log_prob = log_prob.sum(1, keepdim=True)
        # print("log_prob={}".format(log_prob))
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 512)
        self.l5 = nn.Linear(512, 512)
        self.l6 = nn.Linear(512, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, min_action):
        self.alpha = 0.2
        self.max_action = torch.Tensor(max_action).to(device)
        self.min_action = torch.Tensor(min_action).to(device)

        self.actor = Actor(state_dim, action_dim, self.max_action, self.min_action)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def select_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        action, _ = self.actor(state)
        return action.to(device).data.cpu().numpy().flatten()
    
    def train(self, replay_buffer):
        # Update the policy and critic networks
        if len(replay_buffer) > batch_size:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)

            state_batch = torch.FloatTensor(state_batch).to(device)
            action_batch = torch.FloatTensor(action_batch).to(device)
            reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(device)
            done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)

            # Update the critic
            with torch.no_grad():
                next_action_batch, log_prob_next = self.actor(next_state_batch)
                q1_next, q2_next = self.critic(next_state_batch, next_action_batch)
                q_next = torch.min(q1_next, q2_next)
                # print("log_prob_next shape={}".format(log_prob_next.shape))
                # print("q_next shape={}".format(q_next.shape))
                # print("done_batch shape={}".format(done_batch.shape))
                # print("reward_batch shape={}".format(reward_batch.shape))
                # print("state_batch shape={}".format(state_batch.shape))
                # print("action_batch shape={}".format(action_batch.shape))
                target_q_values = reward_batch + gamma * (1 - done_batch.float()) * (q_next - self.alpha * log_prob_next)
            q1, q2 = self.critic(state_batch, action_batch)
            critic_loss = F.mse_loss(q1, target_q_values) + F.mse_loss(q2, target_q_values)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update the policy
            action_batch, log_prob = self.actor(state_batch)
            q1, q2 = self.critic(state_batch, action_batch)
            q = torch.min(q1, q2)
            policy_loss = -(self.alpha*log_prob - q).mean()

            # Optimize the policy
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            print("update")

    # More methods would be needed here for training the model and updating the networks

# The Replay Buffer
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


# Initialize environment, policy, critic, and replay buffer
env_name = "turtlebot3_continuous_sac"
rospy.init_node('turtlebot3_world_sac', anonymous=True, log_level=rospy.WARN)
env=turtlebot_env()
rospy.loginfo("Gym environment done")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_folder_path = os.getcwd()
outdir = current_folder_path + '/training_results/sac_inference_results/'
torch.manual_seed(2)

run_num_pretrained = 1
log_dir = outdir + "sac_logs" + "/log" + str(run_num_pretrained) 
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

agent = SAC(state_dim, action_dim, max_action, min_action)
agent.actor = agent.actor.to(device)
agent.critic = agent.critic.to(device)

# Load the saved model
agent.actor.load_state_dict(torch.load('/home/sezer/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/training_results/offline_sac_results/saved_models/3/DDPG_offline_training_continuous_sac_59999_actor.pth'))
agent.critic.load_state_dict(torch.load('/home/sezer/catkin_ws/src/turtlebot3_ddpg_collision_avoidance/training_results/offline_sac_results/saved_models/3/DDPG_offline_training_continuous_sac_59999_critic.pth'))

# Set the model in evaluation mode
agent.actor.eval()
agent.critic.eval()

replay_buffer = ReplayBuffer(1e9)  # capacity of the replay buffer

num_episodes = 100000
max_episode_steps = 200
batch_size = 64
gamma = 0.99  # discount factor

print("state_dim={}".format(state_dim))
print("action_dim={}".format(action_dim))
print("max_action={}".format(max_action))
print("min_action={}".format(min_action))
print("max_episode_steps={}".format(max_episode_steps))
print("outdir={}".format(outdir))
print("random action={}".format(env.action_space.sample()))

# action = agent.select_action(env.reset())
# print("action={}".format(action))

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
        action = agent.select_action(state)
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

