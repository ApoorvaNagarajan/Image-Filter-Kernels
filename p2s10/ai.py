# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network


class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    bs_X1, bs_X2, next_bs_X1, next_bs_X2, batch_actions, batch_rewards, batch_dones = [], [], [], [], [], [], []
    for i in ind: 
      state_X1, state_X2, next_state_X1, next_state_X2, action, reward, done = self.storage[i]
      bs_X1.append(np.array(state_X1, copy=False))
      bs_X2.append(np.array(state_X2, copy=False))
      next_bs_X1.append(np.array(next_state_X1, copy=False))
      next_bs_X2.append(np.array(next_state_X2, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(bs_X1), np.array(bs_X2), np.array(next_bs_X1), np.array(next_bs_X2), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)
	
start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e5 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated	
min_episode_reward = -10000


class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=(1,1))
    self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=(1,1))
    self.conv3 = nn.Conv2d(32, 16, 3, stride=2, padding=(1,1))
    self.conv4 = nn.Conv2d(16, 10, 3, stride=2, padding=(1,1))
    self.fc1 = nn.Linear(14, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, action_dim)
    self.max_action = max_action

  def forward(self, x1, x2):                # input x1: 32x32x1, x2=3
    h = F.relu(self.conv1(x1))              # 32x32x16
    h = F.relu(self.conv2(h))               # 32x32x32
    h = F.relu(self.conv3(h))               # 16x16x16
    h = F.relu(self.conv4(h))               # 8x8x10
    h = F.avg_pool2d(h, h.size()[2:])       # 10
    h = h.view(-1, 10)
    h = torch.cat([h, x2], dim=1)
    h = F.relu(self.fc1(h))
    h = F.relu(self.fc2(h))
    h = self.max_action * torch.tanh(self.fc3(h))
    return h

class Critic(nn.Module):
  
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=(1,1))
    self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=(1,1))
    self.conv3 = nn.Conv2d(32, 16, 3, stride=2, padding=(1,1))
    self.conv4 = nn.Conv2d(16, 10, 3, stride=2, padding=(1,1))
    self.fc1 = nn.Linear(14 + action_dim, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 1)
    # Defining the second Critic neural network
    self.conv5 = nn.Conv2d(1, 16, 3, stride=1, padding=(1,1))
    self.conv6 = nn.Conv2d(16, 32, 3, stride=1, padding=(1,1))
    self.conv7 = nn.Conv2d(32, 16, 3, stride=2, padding=(1,1))
    self.conv8 = nn.Conv2d(16, 10, 3, stride=2, padding=(1,1))
    self.fc4 = nn.Linear(14 + action_dim, 256)
    self.fc5 = nn.Linear(256, 128)
    self.fc6 = nn.Linear(128, 1)

  def forward(self, x1, x2, u):
    # Forward-Propagation on the first Critic Neural Network
    h1 = F.relu(self.conv1(x1))              # 32x32x16
    h1 = F.relu(self.conv2(h1))               # 32x32x32
    h1 = F.relu(self.conv3(h1))               # 16x16x16
    h1 = F.relu(self.conv4(h1))               # 8x8x10
    h1 = F.avg_pool2d(h1, h1.size()[2:])       # 10
    h1 = h1.view(-1, 10)
    h1 = torch.cat([h1, x2], dim=1)
    h1 = torch.cat([h1, u], dim=1)
    h1 = F.relu(self.fc1(h1))
    h1 = F.relu(self.fc2(h1))
    h1 = self.fc3(h1)
    # Forward-Propagation on the second Critic Neural Network
    h2 = F.relu(self.conv1(x1))              # 32x32x16
    h2 = F.relu(self.conv2(h2))               # 32x32x32
    h2 = F.relu(self.conv3(h2))               # 16x16x16
    h2 = F.relu(self.conv4(h2))               # 8x8x10
    h2 = F.avg_pool2d(h2, h2.size()[2:])       # 10
    h2 = h2.view(-1, 10)
    h2 = torch.cat([h2, x2], dim=1)
    h2 = torch.cat([h2, u], dim=1)
    h2 = F.relu(self.fc1(h2))
    h2 = F.relu(self.fc2(h2))
    h2 = self.fc3(h2)
    return h1, h2

  def Q1(self, x1,x2, u):
    h1 = F.relu(self.conv1(x1))              # 32x32x16
    h1 = F.relu(self.conv2(h1))               # 32x32x32
    h1 = F.relu(self.conv3(h1))               # 16x16x16
    h1 = F.relu(self.conv4(h1))               # 8x8x10
    h1 = F.avg_pool2d(h1, h1.size()[2:])       # 10
    h1 = h1.view(-1, 10)
    h1 = torch.cat([h1, x2], dim=1)
    h1 = torch.cat([h1, u], dim=1)
    h1 = F.relu(self.fc1(h1))
    h1 = F.relu(self.fc2(h1))
    h1 = self.fc3(h1)
    return h1

class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action)
    self.actor_target = Actor(state_dim, action_dim, max_action)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim)
    self.critic_target = Critic(state_dim, action_dim)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action
    self.replay_buffer = ReplayBuffer()
    self.total_timesteps = 0
    self.episode_reward = 0
    self.episode_num = 0
    self.episode_timesteps = 0

  def select_action(self, X1, X2):
        print(X2)
        if(self.total_timesteps < start_timesteps):
            print("random action ", self.total_timesteps)
            return np.random.randint(-5,5, size=1)
        else:
            print("nw action ", self.total_timesteps)
            X1 = torch.Tensor(X1.reshape(1, 1, 32, 32))
            X2 = torch.Tensor(np.asarray(X2).reshape(1, -1))
            return self.actor(Variable(X1, volatile = True), Variable(X2, volatile = True)).cpu().data.numpy().flatten()

  def train(self, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      bs_X1, bs_X2, next_bs_X1, next_bs_X2, batch_actions, batch_rewards, batch_dones = self.replay_buffer.sample(batch_size)
      X1 = Variable(torch.Tensor(bs_X1), volatile = False)
      X2 = Variable(torch.Tensor(bs_X2), volatile = False)
      next_X1 = Variable(torch.Tensor(next_bs_X1), volatile = False)
      next_X2 = Variable(torch.Tensor(next_bs_X2), volatile = False)
      action = Variable(torch.Tensor(batch_actions), volatile = False)
      reward = Variable(torch.Tensor(batch_rewards), volatile = True)
      done = Variable(torch.Tensor(batch_dones), volatile = True)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_X1, next_X2)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = Variable(torch.Tensor(batch_actions), volatile = True).data.normal_(0, policy_noise)
      noise = Variable(noise.clamp(-noise_clip, noise_clip), volatile = True)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_X1, next_X2, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(X1, X2, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      #print("critic loss ",critic_loss)
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(X1, X2, self.actor(X1, X2)).mean()
        #print("actor_loss ",actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
		  
    
  def add_replay_buff(self, X1, X2, new_X1, new_X2, action, reward, done_flag):
        self.episode_reward += reward
        # if reward is lesser than min reward, end the episode
        if(self.episode_reward<min_episode_reward):
            done_flag = 1
        self.replay_buffer.add((X1, X2, new_X1, new_X2, action, reward, done_flag))
        self.total_timesteps += 1
        self.episode_timesteps += 1
        # If episode is done, train the model
        if (done_flag == 1):
            print(self.episode_num, " : EPISODE REWARD ", self.episode_reward)
            self.train(self.episode_timesteps)
            self.episode_reward = 0
            self.episode_num += 1
            self.episode_timesteps = 0
        return done_flag

  # Making a save method to save a trained model
  def save(self):
    torch.save(self.actor.state_dict(), 'last_actor.pth')
    torch.save(self.critic.state_dict(), 'last_critic.pth')
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    if os.path.isfile('last_actor.pth'):
        self.actor.load_state_dict(torch.load('last_actor.pth'))
    if os.path.isfile('last_critic.pth'):
        self.critic.load_state_dict(torch.load('last_critic.pth' ))
	
	