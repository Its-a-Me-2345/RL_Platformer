# RL Platformer

A Gymnasium + Pygame 2D platformer environment, trained end-to-end with Deep Q-Network (DQN) using only pixel observations.


![DC_clip](https://github.com/user-attachments/assets/2d07d1f3-5c3d-4e6c-88c2-1ddc8f57726a)


The trained agent can reliably complete the entire level without failing.

## Features
- Play the level yourself (`game.py`)
- Watch the trained agent play (`test.py`)
- Race head-to-head against the AI (`compete.py` - human in red, agent in green)

## Reward Function

| Condition                                    | Reward                  | Explanation                                                                 |
|----------------------------------------------|-------------------------|-----------------------------------------------------------------------------|
| Every step                                   | `+0.1 × Δdistance - 0.1`| `Δdistance` = reduction in Euclidean distance to goal since last step. Strong forward progress signal + small living penalty |
| No forward movement (stuck)                  | extra `-0.01`           | Penalizes being blocked / zero x-velocity                                   |
| Jumping while already in the air             | `-0.1`                  | Discourages jump-spamming                                                   |
| Falling off the bottom of the world          | `-20`                   | Death penalty → episode terminates                                          |
| Reaching the goal (green block)               | `+100`                  | Success → episode terminates                                                |

## Files

| File              | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `game_env.py`     | Core Gymnasium environment with physics, camera, collision and the complete level |
| `compete.py`      | Human vs Agent race mode (first to the goal wins)                           |
| `game.py`         | Simple human-playable version                                               |
| `test_agent.py`   | Watch the trained agent play                                  |
| `model.pth`       | Trained DQN weights (included in the repo)                                  |

## Observation & Action Space
- Observation: 84×84 grayscale frames, 4-frame stacked 
- Actions (Discrete(2)):
  - 0 → no action
  - 1 → jump (up arrow key)

## Installation

```bash
# Clone or download the repo
git clone https://github.com/Its-a-Me-2345/RL_Platformer.git
cd RL_Platformer

# Recommended: create a virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
# or
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
