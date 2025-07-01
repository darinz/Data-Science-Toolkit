# Python Reinforcement Learning Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Gym](https://img.shields.io/badge/Gym-0.21%2B-blue.svg)](https://gym.openai.com/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-1.5%2B-green.svg)](https://stable-baselines3.readthedocs.io/)

A comprehensive guide to Reinforcement Learning in Python for data science and machine learning applications.

## Table of Contents

1. [Introduction to Reinforcement Learning](#introduction-to-reinforcement-learning)
2. [Q-Learning](#q-learning)
3. [Deep Q-Network (DQN)](#deep-q-network-dqn)
4. [Policy Gradient Methods](#policy-gradient-methods)
5. [Actor-Critic Methods](#actor-critic-methods)
6. [Multi-Agent RL](#multi-agent-rl)
7. [Best Practices](#best-practices)

## Introduction to Reinforcement Learning

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment.

### Key Concepts

- **Agent**: The learner/decision maker
- **Environment**: The world the agent interacts with
- **State**: Current situation of the environment
- **Action**: What the agent can do
- **Reward**: Feedback from the environment
- **Policy**: Strategy for choosing actions

### Basic Setup

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gym
from collections import defaultdict, deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
```

## Q-Learning

### Basic Q-Learning Implementation

```python
class QLearningAgent:
    """Simple Q-Learning agent."""
    
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning update rule."""
        current_q = self.q_table[state][action]
        
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def get_q_table(self):
        """Get the Q-table as a regular dictionary."""
        return dict(self.q_table)

def train_q_learning_agent(env, agent, episodes=1000):
    """Train Q-learning agent."""
    
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    return episode_rewards

# Example: Simple Grid World
class SimpleGridWorld:
    """Simple grid world environment."""
    
    def __init__(self, size=4):
        self.size = size
        self.state = 0
        self.goal = size * size - 1
        
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        x, y = self.state // self.size, self.state % self.size
        
        if action == 0:  # Up
            x = max(0, x - 1)
        elif action == 1:  # Right
            y = min(self.size - 1, y + 1)
        elif action == 2:  # Down
            x = min(self.size - 1, x + 1)
        elif action == 3:  # Left
            y = max(0, y - 1)
        
        self.state = x * self.size + y
        
        # Reward: 1 for reaching goal, -0.1 for each step
        if self.state == self.goal:
            reward = 1
            done = True
        else:
            reward = -0.1
            done = False
        
        return self.state, reward, done, {}

# Train Q-learning agent
env = SimpleGridWorld()
agent = QLearningAgent(state_size=16, action_size=4, learning_rate=0.1, epsilon=0.1)
rewards = train_q_learning_agent(env, agent, episodes=500)

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.title('Q-Learning Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()

# Visualize Q-table
q_table = agent.get_q_table()
q_matrix = np.zeros((16, 4))
for state in range(16):
    if state in q_table:
        q_matrix[state] = q_table[state]

plt.figure(figsize=(12, 8))
sns.heatmap(q_matrix, annot=True, fmt='.2f', cmap='viridis')
plt.title('Q-Table Visualization')
plt.xlabel('Action (0=Up, 1=Right, 2=Down, 3=Left)')
plt.ylabel('State')
plt.show()
```

### Advanced Q-Learning

```python
class AdvancedQLearningAgent(QLearningAgent):
    """Advanced Q-Learning agent with experience replay and target network."""
    
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000):
        super().__init__(state_size, action_size, learning_rate, discount_factor, epsilon)
        
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=memory_size)
        self.batch_size = 32
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Learn from past experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            self.learn(state, action, reward, next_state, done)
    
    def choose_action(self, state):
        """Choose action with decaying epsilon."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_advanced_q_learning(env, agent, episodes=1000):
    """Train advanced Q-learning agent."""
    
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Learn from experience
            agent.replay()
            
            state = next_state
            total_reward += reward
        
        # Decay epsilon
        agent.decay_epsilon()
        
        episode_rewards.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return episode_rewards

# Train advanced Q-learning agent
advanced_agent = AdvancedQLearningAgent(state_size=16, action_size=4)
advanced_rewards = train_advanced_q_learning(env, advanced_agent, episodes=500)

# Compare results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(rewards, label='Basic Q-Learning')
plt.plot(advanced_rewards, label='Advanced Q-Learning')
plt.title('Training Progress Comparison')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), label='Basic Q-Learning')
plt.plot(np.convolve(advanced_rewards, np.ones(50)/50, mode='valid'), label='Advanced Q-Learning')
plt.title('Smoothed Training Progress')
plt.xlabel('Episode')
plt.ylabel('Average Reward (50 episodes)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Deep Q-Network (DQN)

### DQN Implementation

```python
class DQNNetwork(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """Deep Q-Network agent."""
    
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = 32
        
        # Training parameters
        self.update_target_every = 100
        self.step_count = 0
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Learn from past experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.BoolTensor([exp[4] for exp in batch])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn_agent(env, agent, episodes=1000):
    """Train DQN agent."""
    
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Learn from experience
            agent.replay()
            
            state = next_state
            total_reward += reward
        
        # Decay epsilon
        agent.decay_epsilon()
        
        episode_rewards.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return episode_rewards

# Train DQN agent
dqn_agent = DQNAgent(state_size=16, action_size=4)
dqn_rewards = train_dqn_agent(env, dqn_agent, episodes=500)

# Plot DQN results
plt.figure(figsize=(10, 6))
plt.plot(dqn_rewards, label='DQN')
plt.title('DQN Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.show()
```

## Policy Gradient Methods

### REINFORCE Algorithm

```python
class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm."""
    
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.fc3(x))

class REINFORCEAgent:
    """REINFORCE agent implementation."""
    
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def choose_action(self, state):
        """Choose action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_network(state_tensor)
        
        # Sample action from probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item()
    
    def store_transition(self, state, action, reward):
        """Store transition for current episode."""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def update_policy(self):
        """Update policy using REINFORCE algorithm."""
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate discounted returns
        returns = []
        R = 0
        for reward in reversed(self.episode_rewards):
            R = reward + self.discount_factor * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
        
        # Convert to tensors
        states = torch.FloatTensor(self.episode_states)
        actions = torch.LongTensor(self.episode_actions)
        
        # Calculate policy loss
        action_probs = self.policy_network(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        loss = -(log_probs * returns).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

def train_reinforce_agent(env, agent, episodes=1000):
    """Train REINFORCE agent."""
    
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward)
            
            state = next_state
            total_reward += reward
        
        # Update policy
        agent.update_policy()
        
        episode_rewards.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    return episode_rewards

# Train REINFORCE agent
reinforce_agent = REINFORCEAgent(state_size=16, action_size=4)
reinforce_rewards = train_reinforce_agent(env, reinforce_agent, episodes=500)

# Plot REINFORCE results
plt.figure(figsize=(10, 6))
plt.plot(reinforce_rewards, label='REINFORCE')
plt.title('REINFORCE Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.show()
```

## Actor-Critic Methods

### Actor-Critic Implementation

```python
class ActorCriticNetwork(nn.Module):
    """Actor-Critic network architecture."""
    
    def __init__(self, input_size, output_size):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        
        # Actor (policy) head
        self.actor_fc = nn.Linear(64, output_size)
        self.actor_softmax = nn.Softmax(dim=-1)
        
        # Critic (value) head
        self.critic_fc = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Actor output (action probabilities)
        actor_output = self.actor_softmax(self.actor_fc(x))
        
        # Critic output (state value)
        critic_output = self.critic_fc(x)
        
        return actor_output, critic_output

class ActorCriticAgent:
    """Actor-Critic agent implementation."""
    
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.network = ActorCriticNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
    
    def choose_action(self, state):
        """Choose action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self.network(state_tensor)
        
        # Sample action from probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item(), value.item()
    
    def store_transition(self, state, action, reward, value):
        """Store transition for current episode."""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_values.append(value)
    
    def update_policy(self):
        """Update policy using Actor-Critic algorithm."""
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate advantages
        advantages = []
        returns = []
        R = 0
        
        for reward in reversed(self.episode_rewards):
            R = reward + self.discount_factor * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        values = torch.FloatTensor(self.episode_values)
        advantages = returns - values
        
        # Convert to tensors
        states = torch.FloatTensor(self.episode_states)
        actions = torch.LongTensor(self.episode_actions)
        
        # Get current policy and value predictions
        action_probs, value_preds = self.network(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = nn.MSELoss()(value_preds.squeeze(), returns)
        
        # Total loss
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []

def train_actor_critic_agent(env, agent, episodes=1000):
    """Train Actor-Critic agent."""
    
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, value = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, value)
            
            state = next_state
            total_reward += reward
        
        # Update policy
        agent.update_policy()
        
        episode_rewards.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
    
    return episode_rewards

# Train Actor-Critic agent
ac_agent = ActorCriticAgent(state_size=16, action_size=4)
ac_rewards = train_actor_critic_agent(env, ac_agent, episodes=500)

# Plot Actor-Critic results
plt.figure(figsize=(10, 6))
plt.plot(ac_rewards, label='Actor-Critic')
plt.title('Actor-Critic Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.show()
```

## Multi-Agent RL

### Simple Multi-Agent Environment

```python
class MultiAgentGridWorld:
    """Multi-agent grid world environment."""
    
    def __init__(self, size=4, num_agents=2):
        self.size = size
        self.num_agents = num_agents
        self.agents_positions = [0] * num_agents
        self.goals = [size * size - 1] * num_agents
        
    def reset(self):
        self.agents_positions = [0] * self.num_agents
        return self.agents_positions
    
    def step(self, actions):
        rewards = []
        dones = []
        
        for i, action in enumerate(actions):
            x, y = self.agents_positions[i] // self.size, self.agents_positions[i] % self.size
            
            if action == 0:  # Up
                x = max(0, x - 1)
            elif action == 1:  # Right
                y = min(self.size - 1, y + 1)
            elif action == 2:  # Down
                x = min(self.size - 1, x + 1)
            elif action == 3:  # Left
                y = max(0, y - 1)
            
            self.agents_positions[i] = x * self.size + y
            
            # Reward: 1 for reaching goal, -0.1 for each step
            if self.agents_positions[i] == self.goals[i]:
                reward = 1
                done = True
            else:
                reward = -0.1
                done = False
            
            rewards.append(reward)
            dones.append(done)
        
        return self.agents_positions, rewards, any(dones), {}

class MultiAgentQLearning:
    """Multi-agent Q-learning implementation."""
    
    def __init__(self, num_agents, state_size, action_size):
        self.num_agents = num_agents
        self.agents = [QLearningAgent(state_size, action_size) for _ in range(num_agents)]
    
    def choose_actions(self, states):
        """Choose actions for all agents."""
        return [agent.choose_action(state) for agent, state in zip(self.agents, states)]
    
    def learn(self, states, actions, rewards, next_states, dones):
        """Learn for all agents."""
        for i in range(self.num_agents):
            self.agents[i].learn(states[i], actions[i], rewards[i], next_states[i], dones[i])

def train_multi_agent(env, agents, episodes=1000):
    """Train multi-agent system."""
    
    episode_rewards = []
    
    for episode in range(episodes):
        states = env.reset()
        total_rewards = [0] * env.num_agents
        done = False
        
        while not done:
            actions = agents.choose_actions(states)
            next_states, rewards, done, _ = env.step(actions)
            
            agents.learn(states, actions, rewards, next_states, [done] * env.num_agents)
            
            states = next_states
            for i in range(env.num_agents):
                total_rewards[i] += rewards[i]
        
        episode_rewards.append(sum(total_rewards))
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Average Total Reward: {avg_reward:.2f}")
    
    return episode_rewards

# Train multi-agent system
multi_env = MultiAgentGridWorld(size=4, num_agents=2)
multi_agents = MultiAgentQLearning(num_agents=2, state_size=16, action_size=4)
multi_rewards = train_multi_agent(multi_env, multi_agents, episodes=500)

# Plot multi-agent results
plt.figure(figsize=(10, 6))
plt.plot(multi_rewards, label='Multi-Agent Q-Learning')
plt.title('Multi-Agent Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward (All Agents)')
plt.legend()
plt.grid(True)
plt.show()
```

## Best Practices

### Complete RL Pipeline

```python
class RLExperiment:
    """Complete reinforcement learning experiment framework."""
    
    def __init__(self):
        self.results = {}
        
    def run_experiment(self, env, agent_type, agent_params, episodes=1000):
        """Run a complete RL experiment."""
        
        print(f"Running experiment with {agent_type}")
        
        if agent_type == 'q_learning':
            agent = QLearningAgent(**agent_params)
            rewards = train_q_learning_agent(env, agent, episodes)
        elif agent_type == 'dqn':
            agent = DQNAgent(**agent_params)
            rewards = train_dqn_agent(env, agent, episodes)
        elif agent_type == 'reinforce':
            agent = REINFORCEAgent(**agent_params)
            rewards = train_reinforce_agent(env, agent, episodes)
        elif agent_type == 'actor_critic':
            agent = ActorCriticAgent(**agent_params)
            rewards = train_actor_critic_agent(env, agent, episodes)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.results[agent_type] = {
            'rewards': rewards,
            'final_avg_reward': np.mean(rewards[-100:]),
            'best_episode': np.argmax(rewards),
            'best_reward': np.max(rewards)
        }
        
        return agent, rewards
    
    def compare_algorithms(self, env, episodes=1000):
        """Compare different RL algorithms."""
        
        agent_configs = {
            'Q-Learning': {
                'agent_type': 'q_learning',
                'agent_params': {'state_size': 16, 'action_size': 4, 'learning_rate': 0.1}
            },
            'DQN': {
                'agent_type': 'dqn',
                'agent_params': {'state_size': 16, 'action_size': 4, 'learning_rate': 0.001}
            },
            'REINFORCE': {
                'agent_type': 'reinforce',
                'agent_params': {'state_size': 16, 'action_size': 4, 'learning_rate': 0.001}
            },
            'Actor-Critic': {
                'agent_type': 'actor_critic',
                'agent_params': {'state_size': 16, 'action_size': 4, 'learning_rate': 0.001}
            }
        }
        
        for name, config in agent_configs.items():
            print(f"\n{'='*50}")
            print(f"TRAINING {name.upper()}")
            print(f"{'='*50}")
            
            self.run_experiment(env, config['agent_type'], config['agent_params'], episodes)
    
    def plot_comparison(self):
        """Plot comparison of all algorithms."""
        
        plt.figure(figsize=(15, 10))
        
        # Plot training curves
        plt.subplot(2, 2, 1)
        for name, result in self.results.items():
            plt.plot(result['rewards'], label=name, alpha=0.7)
        plt.title('Training Progress Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        
        # Plot final performance
        plt.subplot(2, 2, 2)
        names = list(self.results.keys())
        final_rewards = [self.results[name]['final_avg_reward'] for name in names]
        plt.bar(names, final_rewards)
        plt.title('Final Average Reward (Last 100 Episodes)')
        plt.ylabel('Average Reward')
        plt.xticks(rotation=45)
        
        # Plot best performance
        plt.subplot(2, 2, 3)
        best_rewards = [self.results[name]['best_reward'] for name in names]
        plt.bar(names, best_rewards)
        plt.title('Best Episode Reward')
        plt.ylabel('Best Reward')
        plt.xticks(rotation=45)
        
        # Plot convergence
        plt.subplot(2, 2, 4)
        for name, result in self.results.items():
            smoothed = np.convolve(result['rewards'], np.ones(50)/50, mode='valid')
            plt.plot(smoothed, label=name, alpha=0.7)
        plt.title('Smoothed Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive experiment report."""
        
        report = "=== REINFORCEMENT LEARNING EXPERIMENT REPORT ===\n"
        
        for name, result in self.results.items():
            report += f"\n--- {name.upper()} RESULTS ---\n"
            report += f"Final Average Reward: {result['final_avg_reward']:.4f}\n"
            report += f"Best Episode: {result['best_episode']}\n"
            report += f"Best Reward: {result['best_reward']:.4f}\n"
        
        return report

# Run complete experiment
experiment = RLExperiment()
experiment.compare_algorithms(env, episodes=500)
experiment.plot_comparison()

# Generate report
report = experiment.generate_report()
print(report)
```

## Summary

Reinforcement Learning provides powerful tools for decision-making:

- **Q-Learning**: Value-based method for discrete action spaces
- **DQN**: Deep Q-learning for complex environments
- **Policy Gradient**: Direct policy optimization
- **Actor-Critic**: Combines policy and value function learning
- **Multi-Agent RL**: Learning in multi-agent environments
- **Best Practices**: Systematic experimentation and evaluation

Mastering reinforcement learning will help you build intelligent agents that can learn and adapt.

## Next Steps

- Practice with more complex environments
- Explore advanced RL algorithms
- Learn about continuous control problems
- Study real-world RL applications

---

**Happy Reinforcement Learning!** 🤖✨ 