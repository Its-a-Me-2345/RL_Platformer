import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import deque
import random
import os
import copy
import math
import cv2

from tqdm import tqdm
from game_env import PlatformerLongEnv


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

LR = 1e-5
GAMMA = 0.99
TAU = 1e-4

BUFFER_SIZE = 100000
BATCH_SIZE = 256

UPDATE_EVERY = 4
TARGET_UPDATE_EVERY = 10000

SEED = 69
NUM_FRAMES = 4

EPS_START = 0.11
EPS_END = 0.02
EPS_DECAY = 500000


class PreprocessingWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super(PreprocessingWrapper, self).__init__(env)
        self.shape = (shape[1], shape[0])

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shape[0], shape[1], 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1).astype(np.uint8)


class FrameStackWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_frames=NUM_FRAMES):
        super(FrameStackWrapper, self).__init__(env)
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

        low = np.repeat(self.observation_space.low, num_frames, axis=-1)
        high = np.repeat(self.observation_space.high, num_frames, axis=-1)

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(
                self.observation_space.shape[0],
                self.observation_space.shape[1],
                num_frames
            ),
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        for _ in range(self.num_frames):
            self.frames.append(obs)

        return self._get_obs(), info

    def observation(self, obs):
        self.frames.append(obs)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.stack([e[0] for e in experiences if e is not None])
        actions = np.vstack([e[1] for e in experiences if e is not None]).astype(np.int64)
        rewards = np.vstack([e[2] for e in experiences if e is not None])
        next_states = np.stack([e[3] for e in experiences if e is not None])
        dones = np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)

        states = torch.from_numpy(states).float().permute(0, 3, 1, 2).to(DEVICE) / 255.0
        actions = torch.from_numpy(actions).long().to(DEVICE)
        rewards = torch.from_numpy(rewards).float().to(DEVICE)
        next_states = torch.from_numpy(next_states).float().permute(0, 3, 1, 2).to(DEVICE) / 255.0
        dones = torch.from_numpy(dones).float().to(DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_shape, action_size, seed, fc_layers=[512]):
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        H, W, C = state_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten()
        )

        self.flattened_size = self._get_flattened_size(state_shape)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, fc_layers[0]),
            nn.ReLU(),
            nn.Linear(fc_layers[0], action_size)
        )

    def _get_flattened_size(self, state_shape):
        H, W, C = state_shape
        x = torch.zeros(1, C, H, W)
        return self.cnn(x).numel()

    def forward(self, state):
        features = self.cnn(state)
        return self.fc(features)


class DQNAgent:
    def __init__(self, state_shape, action_size, seed):
        self.state_shape = state_shape
        self.action_size = action_size

        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_shape, action_size, seed).to(DEVICE)
        self.qnetwork_target = QNetwork(state_shape, action_size, seed).to(DEVICE)

        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        self.t_step = 0
        self.total_steps = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        self.total_steps += 1

    def act(self, state, eps=0.):
        if random.random() > eps:
            state_tensor = (
                torch.from_numpy(state)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(DEVICE) / 255.0
            )

            self.qnetwork_local.eval()

            with torch.no_grad():
                action_values = self.qnetwork_local(state_tensor)

            self.qnetwork_local.train()

            return np.argmax(action_values.cpu().data.numpy())

        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            Q_targets_next = self.qnetwork_target(next_states).max(dim=1, keepdim=True)[0]

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        del experiences

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def save_checkpoint(self, path='checkpoint_dqn.pth'):
        torch.save(self.qnetwork_local.state_dict(), path)

    def load_checkpoint(self, path):
        if os.path.exists(path):
            self.qnetwork_local.load_state_dict(torch.load(path, map_location=DEVICE))
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            print(f"Loaded model from {path}")
        else:
            print(f"Checkpoint not found: {path}")


def train_dqn(agent, env, n_episodes=5000, max_t=3000):
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in tqdm(range(1, n_episodes + 1), desc="Training DQN Agent"):
        state, _ = env.reset()
        score = 0

        eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * agent.total_steps / EPS_DECAY)

        for t in range(max_t):
            discrete_action = agent.act(state, eps)

            next_state, reward, terminated, truncated, _ = env.step(discrete_action)

            done = terminated or truncated

            agent.step(state, discrete_action, reward, next_state, done)

            state = next_state
            score += reward

            if done:
                break

        scores_deque.append(score)
        scores.append(score)

        avg_score = np.mean(scores_deque)

        if i_episode % 10 == 0:
            tqdm.write(
                f"\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}"
                f"\tEpsilon: {eps:.4f} \tSteps: {agent.total_steps}"
            )

        if i_episode % 20 == 0:
            agent.save_checkpoint(f'retrain_dqn_ep{i_episode}.pth')

    print("\nTraining complete.")

    agent.save_checkpoint('final_checkpoint_dqn1.pth')

    return scores


if __name__ == '__main__':
    env = PlatformerLongEnv(render_mode="rgb_array")

    env = PreprocessingWrapper(env, shape=(84, 84))
    env = FrameStackWrapper(env, num_frames=NUM_FRAMES)

    state_shape = env.observation_space.shape
    action_size = env.action_space.n

    print(f"State shape: {state_shape}, Action size: {action_size}")

    agent = DQNAgent(state_shape=state_shape, action_size=action_size, seed=SEED)

    CHECKPOINT_PATH = 'final_model.pth'
    agent.load_checkpoint(CHECKPOINT_PATH)

    train_dqn(agent, env, n_episodes=5000, max_t=3000)

    env.close()