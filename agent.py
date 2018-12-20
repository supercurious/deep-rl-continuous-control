import numpy as np
import random
import copy
from collections import deque

from neuralnets import Actor, Critic, TwinCritics

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Agent based on DDPG / TD3"""
    
    def __init__(self, state_size, action_size, random_seed, memory,
                 batch_size=256, 
                 gamma=0.99, 
                 tau=1e-3, 
                 LR_actor=1e-3, 
                 LR_critic=1e-3, 
                 weight_decay=0.0001, 
                 num_agents=1, 
                 update_every=20, 
                 num_updates=10, 
                 doubleQ=True, 
                 delay_policy=2, 
                 smooth_policy=0,
                 nstep=3, 
                 grad_steps=1):
        """
        # Parameters
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            memory (object): ReplayBuffer
            batch_size (int): number of experience samples per gradient step
            gamma (float): discount rate
            tau (float): soft update rate
            LR_actor (float): actor learning rate
            LR_critic (float): critic learning rate
            weight_decay (float): critic optimizer weight decay
            num_agents (int): number of parallel agents
            update_every (int): update nets how often?
            num_updates (int): learn from how many minibatches each time?
            doubleQ (bool): use clipped double q-learning?
            delay_policy (int): how many critic steps per actor step?
            smooth_policy (float): scale noise for target policy smoothing
            nstep=3 (int): length of trajectory to sample reward
            grad_steps (int): how many critic gradient steps per minibatch

        """
        self.name = 'agent'
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents 
        self.gamma = gamma
        self.tau = tau
        self.LR_actor = LR_actor
        self.LR_critic = LR_critic
        self.weight_decay = weight_decay
        self.smooth_policy = smooth_policy
        self.doubleQ = doubleQ
        self.nstep = nstep
        #self.noise = [OUNoise(action_size, random.randint(0,2**32)) for _ in range(num_agents)]
        self.memory = memory
        self.batch_size = batch_size
        self.t = 0
        self.t_ep = 0
        self.update_every = update_every  
        self.num_updates = num_updates   
        self.delay_policy = delay_policy  
        self.t_learn = 0
        self.grad_steps = grad_steps

        # Deques to storing n-step trajectory
        self.rewards_nstep = [deque(maxlen=nstep) for _ in range(num_agents)]
        self.states_mem = [deque(maxlen=nstep) for _ in range(num_agents)]
        self.actions_mem = [deque(maxlen=nstep) for _ in range(num_agents)]

        # Log losses
        self.critic_loss_log = {}
        self.actor_loss_log = {}

        # Initialize actor, critic, and target nets
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_actor)
        if self.doubleQ:
            self.critic_local = TwinCritics(state_size, action_size, random_seed).to(device)
            self.critic_target = TwinCritics(state_size, action_size, random_seed).to(device)
        else:
            self.critic_local = Critic(state_size, action_size, random_seed).to(device)
            self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), 
                                           lr=LR_critic, weight_decay=weight_decay)

    def step(self, states, actions, rewards, next_states, dones):
        """Add experience to memory and sample from buffer to learn"""
        self.t += 1  # total time step counter
        self.t_ep += 1  # episide time step counter

        # Store states, actions, rewards over n-step trajectory
        for i in range(self.num_agents):
            self.rewards_nstep[i].append(rewards[i])
            self.states_mem[i].append(states[i])
            self.actions_mem[i].append(actions[i])

        if self.t_ep > self.nstep - 1:
            # Sum discounted rewards over n-step trajectory
            rewards_sum = []
            for reward_deque in self.rewards_nstep:
                rewards_sum.append(sum([(self.gamma**i) * r for i, r in enumerate(reward_deque)]))
            # Starting state and action in n-step trajectory
            states_mem_0 = [self.states_mem[i][0] for i in range(self.num_agents)]
            actions_mem_0 = [self.actions_mem[i][0] for i in range(self.num_agents)]

            # Add experience to memory buffer
            for s, a, r, next_s, d in zip(states_mem_0, actions_mem_0, rewards_sum, next_states, dones):
                self.memory.add(s, a, r, next_s, d)
        
        if dones[0] == True:
            self.t_ep = 0  # reset episode time

        # Sample from replay buffer and learn
        if len(self.memory) > self.batch_size:
            if self.t % self.update_every == 0:
                self.critic_loss_log[self.t] = []  # for logging
                self.actor_loss_log[self.t] = []  # for logging
                for _ in range(self.num_updates):
                    experiences = self.memory.sample()
                    for _ in range(self.grad_steps):
                        self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Get action from policy based on state"""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        # Action space noise
        if add_noise:
            # Add OU noise from unique process for each agent
            #for i in range(len(self.noise)):
            #    action[i] += self.noise[i].sample()

            # Gaussian noise, unique for each agent
            std_dev = 0.1
            for i in range(self.num_agents):
                action += np.random.normal(0, std_dev)
        return np.clip(action, -1, 1) 

    def reset(self):
        for noise in self.noise:
            noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using batch of experience tuples
        # Parameters
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        self.t_learn += 1  # increment learning step counter

        # UPDATE CRITIC
        # Get predicted next-state actions from target net for TD target
        actions_next = self.actor_target(next_states)
        # Add target policy smoothing noise
        noise = (torch.randn_like(actions_next) * self.smooth_policy).clamp(-0.25, 0.25)
        actions_next = (actions_next + noise).clamp(-1, 1)

        # Compute TD target 
        if self.doubleQ:
            Q_targets_next, Q2_targets_next = self.critic_target(next_states, actions_next)
            minQ_targets_next = torch.min(Q_targets_next, Q2_targets_next)
            Q_targets = rewards + (gamma**self.nstep * minQ_targets_next * (1 - dones))
        else:
            Q_targets_next = self.critic_target(next_states, actions_next)
            Q_targets = rewards + (gamma**self.nstep * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        if self.doubleQ:
            Q_expected, Q2_expected = self.critic_local(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets) + F.mse_loss(Q2_expected, Q_targets)
        else:
            Q_expected = self.critic_local(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_loss_log[self.t].append(critic_loss.data.numpy())  # log

        # Minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # gradient clipping
        self.critic_optimizer.step()

        # Delayed policy update
        if self.t_learn % self.delay_policy == 0: 
            # UPDATE ACTOR
            actions_pred = self.actor_local(states)
            if self.doubleQ:
                actor_loss = -self.critic_local.single(states, actions_pred).mean()
            else:
                actor_loss = -self.critic_local(states, actions_pred).mean()
            self.actor_loss_log[self.t].append(actor_loss.data.numpy())  # log

            # Maximize return
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # UPDATE TARGET NETS
            self.soft_update(self.critic_local, self.critic_target, self.tau)
            self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target

        # Parameters
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process"""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        #self.seed = random.seed(seed)
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.normal(0,0.3) for i in range(len(x))])
        self.state = x + dx
        return self.state