import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2
import time
from collections import deque
from game_env import PlatformerLongEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MODEL_PATH = f'final_model.pth'

N_EPISODES = 3

NUM_FRAMES = 4 

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class PreprocessingWrapper(gym.ObservationWrapper):
    """
    Converts (600, 800, 3) to (84, 84, 1)
    """
    def __init__(self, env, shape=(84, 84)):
        super(PreprocessingWrapper, self).__init__(env)
        self.shape = (shape[1], shape[0])
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
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
            low=low, high=high,
            shape=(self.observation_space.shape[0], self.observation_space.shape[1], num_frames),
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

def act(state, qnetwork):
    # Convert state (H, W, C) to Tensor (1, C, H, W) and normalize
    state_tensor = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
    
    qnetwork.eval()
    with torch.no_grad():
        action_values = qnetwork(state_tensor)
    
    return np.argmax(action_values.cpu().data.numpy())

if __name__ == '__main__':
    
    env = PlatformerLongEnv(render_mode="human")
    
    env = PreprocessingWrapper(env, shape=(84, 84))
    env = FrameStackWrapper(env, num_frames=NUM_FRAMES)
    
    state_shape= env.observation_space.shape
    action_size= env.action_space.n
    print(f"State shape: {state_shape}, Action size: {action_size}")

    qnetwork = QNetwork(state_shape=state_shape, action_size=action_size, seed=SEED).to(DEVICE)

    try:
        qnetwork.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Successfully loaded model weights from {MODEL_PATH}")

    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please check the path and ensure the model was trained with frame stacking.")
        env.close()
        exit()
    except Exception as e:
        print(f"Error loading model weights: {e}")
        env.close()
        exit()

    total_scores = []

    for i_episode in range(1, 3):
        state, _ = env.reset()
        score = 0
        done = False
        
        print(f"\nStarting Test Episode {i_episode} ")
        
        while not done:
            discrete_action= act(state, qnetwork)
            
            next_state, reward, terminated, truncated, _ =env.step(discrete_action)

            env.render()

            done= terminated or truncated

            state= next_state
            score+= reward
            
            time.sleep(1/60)

            if done:
                break
        
        total_scores.append(score)
        print(f"Episode {i_episode} Finished! Score: {score}")

    env.close()
    
    print("\nTESTING COMPLETE")
    print(f"Average Score over {N_EPISODES} episodes: {np.mean(total_scores):.2f}")
    print(f"Min Score: {np.min(total_scores):.2f}")
    print(f"Max Score: {np.max(total_scores):.2f}")