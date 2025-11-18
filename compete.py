import pygame
import torch
import numpy as np
import random
from game_env import PlatformerLongEnv, PLATFORMS, PLAYER_SIZE, GOAL_POS, GOAL_COLOR
from test_agent import QNetwork, PreprocessingWrapper, FrameStackWrapper, act, DEVICE, MODEL_PATH, NUM_FRAMES, SEED

HUMAN_COLOR = (231, 76, 60)
AGENT_COLOR = (46, 204, 113)

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("RL Agent vs Human")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28)

    env_human = PlatformerLongEnv(render_mode=None)
    env_human.player_x, env_human.player_y = 40, 530

    env_agent = PlatformerLongEnv(render_mode=None)
    env_agent = PreprocessingWrapper(env_agent, shape=(84, 84))
    env_agent = FrameStackWrapper(env_agent, num_frames=NUM_FRAMES)
    state, _ = env_agent.reset()

    state_shape = env_agent.observation_space.shape
    action_size = env_agent.action_space.n
    qnetwork = QNetwork(state_shape=state_shape, action_size=action_size, seed=SEED).to(DEVICE)
    qnetwork.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    qnetwork.eval()

    running = True
    camera_x = 0
    winner = None
    freeze_agent = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if winner:
            screen.fill((236, 240, 241))
            for (px, py, pw, ph) in PLATFORMS:
                rect = pygame.Rect(int(px - camera_x), py, pw, ph)
                if rect.right >= 0 and rect.left <= 800:
                    pygame.draw.rect(screen, (44, 62, 80), rect)
            goal_rect = pygame.Rect(int(GOAL_POS[0] - camera_x), GOAL_POS[1], *PLAYER_SIZE)
            pygame.draw.rect(screen, GOAL_COLOR, goal_rect)
            human_rect = pygame.Rect(int(env_human.player_x - camera_x), int(env_human.player_y), *PLAYER_SIZE)
            pygame.draw.rect(screen, HUMAN_COLOR, human_rect)
            agent_x = env_agent.env.env.player_x
            agent_y = env_agent.env.env.player_y
            agent_rect = pygame.Rect(int(agent_x - camera_x), int(agent_y), *PLAYER_SIZE)
            pygame.draw.rect(screen, AGENT_COLOR, agent_rect)
            win_text = font.render(f"{winner} Wins!", True, (0, 0, 0))
            screen.blit(win_text, (screen.get_width() // 2 - win_text.get_width() // 2, 20))
            pygame.display.flip()
            pygame.time.wait(3000)
            running = False
            continue

        keys = pygame.key.get_pressed()
        human_action = 3
        move = None
        if keys[pygame.K_LEFT]:
            move = 0
        elif keys[pygame.K_UP]:
            move = 1
        if keys[pygame.K_SPACE]:
            human_action = 2
        elif move is not None:
            human_action = move

        _, _, human_terminated, human_truncated, _ = env_human.step(human_action)
        env_human._update_camera()
        camera_x = env_human.camera_x

        if human_terminated or human_truncated:
            env_human.reset()

        agent_terminated = False
        agent_truncated = False
        if not freeze_agent:
            action = act(state, qnetwork)
            next_state, _, agent_terminated, agent_truncated, _ = env_agent.step(action)
            state = next_state
            if agent_terminated or agent_truncated:
                agent_x = env_agent.env.env.player_x
                if agent_x >= GOAL_POS[0]:
                    winner = "Agent"
                    freeze_agent = True
                    continue
                else:
                    state, _ = env_agent.reset()

        if not winner and env_human.player_x >= GOAL_POS[0]:
            winner = "Human"
            freeze_agent = True

        screen.fill((236, 240, 241))
        for (px, py, pw, ph) in PLATFORMS:
            rect = pygame.Rect(int(px - camera_x), py, pw, ph)
            if rect.right >= 0 and rect.left <= 800:
                pygame.draw.rect(screen, (44, 62, 80), rect)
        goal_rect = pygame.Rect(int(GOAL_POS[0] - camera_x), GOAL_POS[1], *PLAYER_SIZE)
        pygame.draw.rect(screen, GOAL_COLOR, goal_rect)
        human_rect = pygame.Rect(int(env_human.player_x - camera_x), int(env_human.player_y), *PLAYER_SIZE)
        pygame.draw.rect(screen, HUMAN_COLOR, human_rect)
        agent_x = env_agent.env.env.player_x
        agent_y = env_agent.env.env.player_y
        agent_rect = pygame.Rect(int(agent_x - camera_x), int(agent_y), *PLAYER_SIZE)
        pygame.draw.rect(screen, AGENT_COLOR, agent_rect)
        distance = agent_x - env_human.player_x
        dist_text = font.render(f"Distance from agent: {int(distance)}", True, (0, 0, 0))
        screen.blit(dist_text, (20, 20))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()

