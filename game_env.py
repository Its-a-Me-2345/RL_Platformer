import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import math

WIDTH, HEIGHT = 800, 600
FPS = 60
PLAYER_SIZE = (30, 30)
PLAYER_COLOR = (52, 152, 219)
GOAL_COLOR = (39, 174, 96)
PLATFORM_COLOR = (44, 62, 80)
BG_COLOR = (236, 240, 241)
gravity = 0.5
jump_strength = 15
move_speed = 5
max_fall = 15

PLAYER_START = (40, 530)
GOAL_POS = (10440, HEIGHT - 440 + 20 + 40, 100, 20)

PLATFORMS = []
PLATFORMS.append((0, HEIGHT - 40, 1000, 40))
PLATFORMS.append((1100, HEIGHT - 40, 350, 40))
PLATFORMS.append((1600, HEIGHT - 40, 300, 40))

stair_x = 1950
for i in range(5):
    PLATFORMS.append((stair_x + i * 150, HEIGHT - 80 - i * 60, 120, 20))

PLATFORMS += [
    (2750, HEIGHT - 40, 400, 40),
    (2950, HEIGHT - 300, 120, 20),
    (3200, HEIGHT - 40, 200, 40),
    (3500, HEIGHT - 40, 300, 40),
    (3900, HEIGHT - 40, 150, 40),
    (4100, HEIGHT - 90, 100, 20),
    (4300, HEIGHT - 140, 100, 20),
    (4500, HEIGHT - 120, 150, 30),
    (4750, HEIGHT - 260, 120, 20),
    (4950, HEIGHT - 300, 120, 20),
    (5150, HEIGHT - 340, 120, 20),
    (5350, HEIGHT - 220, 120, 20),
    (5550, HEIGHT - 170, 120, 20),
    (5700, HEIGHT - 70, 120, 20),
    (5850, HEIGHT - 150, 50, 20),
    (6000, HEIGHT - 270, 50, 20),
    (6150, HEIGHT - 190, 50, 20),
    (6300, HEIGHT - 250, 50, 20),
    (6450, HEIGHT - 170, 50, 20),
    (5750, HEIGHT - 40, 80, 20),
    (6710, HEIGHT - 80, 120, 20),
    (7000, 400, 120, 20),
    (7200, 200, 120, 20),
    (7200, 100, 120, 20),
]

tower_x = 7450
for i in range(6):
    PLATFORMS.append((tower_x, HEIGHT - 80 - i * 100, 120, 20))

for i in range(10):
    PLATFORMS.append((7750 + i * 80 - 60, HEIGHT - 40, 70, 20))

PLATFORMS += [
    (8600, HEIGHT - 70, 100, 20),
    (8750, HEIGHT - 200, 100, 20),
    (8920, HEIGHT - 140, 100, 20),
    (9090, HEIGHT - 240, 100, 20),
    (9220, 300, 100, 20),
    (9050, 350, 100, 20),
    (8880, 380, 100, 20),
    (9400, HEIGHT - 150, 40, 15),
    (9520, HEIGHT - 270, 40, 15),
    (9640, HEIGHT - 190, 40, 15),
    (9760, HEIGHT - 250, 40, 15),
    (9920, HEIGHT - 170, 40, 15),
    (10050, HEIGHT - 80, 100, 20),
    (10180, HEIGHT - 200, 100, 20),
    (10310, HEIGHT - 320, 100, 20),
]

WORLD_MIN_X = 0
WORLD_MAX_X = 12000
CAMERA_LEFT_MARGIN = WIDTH // 3
CAMERA_RIGHT_MARGIN = WIDTH - WIDTH // 3

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

class PlatformerLongEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        pygame.font.init()
        self._offscreen = pygame.Surface((WIDTH, HEIGHT))
        self.font = pygame.font.SysFont(None, 32)
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("RL Platformer - Long Level")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.player_x, self.player_y = float(PLAYER_START[0]), float(PLAYER_START[1])
        self.velocity = [0.0, 0.0]
        self.onGround = False
        self.camera_x = 0.0
        return self._get_obs(), {}

    def _update_camera(self):
        player_screen_x = self.player_x - self.camera_x
        if player_screen_x < CAMERA_LEFT_MARGIN:
            self.camera_x = clamp(self.player_x - CAMERA_LEFT_MARGIN, WORLD_MIN_X, WORLD_MAX_X - WIDTH)
        elif player_screen_x > CAMERA_RIGHT_MARGIN:
            self.camera_x = clamp(self.player_x - CAMERA_RIGHT_MARGIN, WORLD_MIN_X, WORLD_MAX_X - WIDTH)

    def _move_and_collide(self, dx, dy):
        x = self.player_x + dx
        y = self.player_y
        player_rect_h = pygame.Rect(int(x), int(y), PLAYER_SIZE[0], PLAYER_SIZE[1])
        for (px, py, pw, ph) in PLATFORMS:
            plat = pygame.Rect(px, py, pw, ph)
            if player_rect_h.colliderect(plat):
                if dx > 0:
                    x = plat.left - PLAYER_SIZE[0]
                elif dx < 0:
                    x = plat.right
                player_rect_h.x = int(x)
        y += dy
        player_rect_v = pygame.Rect(int(x), int(y), PLAYER_SIZE[0], PLAYER_SIZE[1])
        on_ground = False
        for (px, py, pw, ph) in PLATFORMS:
            plat = pygame.Rect(px, py, pw, ph)
            if player_rect_v.colliderect(plat):
                if dy > 0:
                    y = plat.top - PLAYER_SIZE[1]
                    self.velocity[1] = 0.0
                    on_ground = True
                elif dy < 0:
                    y = plat.bottom
                    self.velocity[1] = 0.0
                player_rect_v.y = int(y)
        return x, y, on_ground

    def _draw(self, surface):
        surface.fill(BG_COLOR)
        for (px, py, pw, ph) in PLATFORMS:
            rect = pygame.Rect(int(px - self.camera_x), py, pw, ph)
            if rect.right >= 0 and rect.left <= WIDTH:
                pygame.draw.rect(surface, PLATFORM_COLOR, rect)
        goal_rect = pygame.Rect(int(GOAL_POS[0] - self.camera_x), GOAL_POS[1], *PLAYER_SIZE)
        pygame.draw.rect(surface, GOAL_COLOR, goal_rect)
        player_rect = pygame.Rect(int(self.player_x - self.camera_x), int(self.player_y), *PLAYER_SIZE)
        pygame.draw.rect(surface, PLAYER_COLOR, player_rect)
        if self.font:
            surface.blit(self.font.render("Long Level", True, (50, 50, 50)), (20, 20))

    def _get_obs(self):
        self._draw(self._offscreen)
        arr = pygame.surfarray.array3d(self._offscreen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def step(self, action):
        prev_x, prev_y = self.player_x, self.player_y
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return self._get_obs(), 0, True, False, {}
        dx = move_speed
        if action == 1 and self.onGround:
            self.velocity[1] = -jump_strength
            self.onGround = False

        self.velocity[1] += gravity
        if self.velocity[1] > max_fall:
            self.velocity[1] = max_fall
        new_x, new_y, grounded = self._move_and_collide(dx, self.velocity[1])
        self.player_x = clamp(new_x, WORLD_MIN_X, WORLD_MAX_X - PLAYER_SIZE[0])
        self.player_y = new_y
        self.onGround = grounded
        self._update_camera()
        prev_dist = math.hypot(prev_x - GOAL_POS[0], prev_y - GOAL_POS[1])
        curr_dist = math.hypot(self.player_x - GOAL_POS[0], self.player_y - GOAL_POS[1])
        dist_delta = prev_dist - curr_dist
        reward = dist_delta * 0.1 - 0.1
        if self.player_x == prev_x:
            reward -= 0.01
        terminated = False
        if self.onGround == False and action == 1:
            reward -= 0.1
        if self.player_y > HEIGHT + 300:
            reward = -20
            terminated = True
        player_rect = pygame.Rect(int(self.player_x), int(self.player_y), *PLAYER_SIZE)
        goal_rect = pygame.Rect(GOAL_POS[0], GOAL_POS[1], *PLAYER_SIZE)
        if player_rect.colliderect(goal_rect):
            reward = 100
            terminated = True
        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    def render(self):
        if self.render_mode == "human":
            self._draw(self.screen)
            pygame.display.flip()
            self.clock.tick(FPS)
        else:
            return self._get_obs()

    def close(self):
        if self.screen:
            pygame.quit()

try:
    from gymnasium.envs.registration import register
    register(id="PlatformerLong-v0", entry_point="platformer_long_env:PlatformerLongEnv")
except Exception:
    pass
