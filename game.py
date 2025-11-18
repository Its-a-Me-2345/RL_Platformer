import pygame
import sys
import json
import time

# Game Configs
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

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Platformer - Single Long Level")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 32)

playerStart = (8330, 530)  # NOTE: HEIGHT isn't defined yet below; use 600 to match later

# -- build the long world (same as your list)
platforms_world = []
platforms_world.append((0, 600 - 40, 1000, 40))
platforms_world.append((1100, 600 - 40, 350, 40))
platforms_world.append((1600, 600 - 40, 300, 40))
stair_x = 1950
for i in range(5):
    platforms_world.append((stair_x + i * 150, 600 - 80 - i * 60, 120, 20))
platforms_world.append((2750, 600 - 40, 400, 40))
platforms_world.append((2950, 600 - 300, 120, 20))
platforms_world.append((3200, 600 - 40, 200, 40))
platforms_world.append((3500, 600 - 40, 300, 40))
platforms_world.append((3900, 600 - 40, 150, 40))
platforms_world.append((4100, 600 - 90, 100, 20))
platforms_world.append((4300, 600 - 140, 100, 20))
platforms_world.append((4500, 600 - 120, 150, 30))
platforms_world.append((4750, 600 - 260, 120, 20))
platforms_world.append((4950, 600 - 300, 120, 20))
platforms_world.append((5150, 600 - 340, 120, 20))
platforms_world.append((5350, 600 - 220, 120, 20))
platforms_world.append((5550, 600 - 170, 120, 20))
platforms_world.append((5700, 600 - 70, 120, 20))
platforms_world.append((5850, 600 - 150, 50, 20))
platforms_world.append((6000, 600 - 270, 50, 20))
platforms_world.append((6150, 600 - 190, 50, 20))
platforms_world.append((6300, 600 - 250, 50, 20))
platforms_world.append((6450, 600 - 170, 50, 20))
platforms_world.append((5750, 600 - 40, 80, 20))
platforms_world.append((6650, 600 - 80, 120, 20))
platforms_world.append((6950, 400, 120, 20))
platforms_world.append((7200, 200, 120, 20))
platforms_world.append((7200, 100, 120, 20))
tower_x = 7450
for i in range(6):
    platforms_world.append((tower_x, 600 - 80 - i * 100, 120, 20))
for i in range(10):
    platforms_world.append((7750 + i * 80, 600 - 40, 70, 20))
platforms_world.append((8600, 600 - 70, 100, 20))
platforms_world.append((8750, 600 - 200, 100, 20))
platforms_world.append((8920, 600 - 140, 100, 20))
platforms_world.append((9090, 600 - 240, 100, 20))
platforms_world.append((9260, 300, 100, 20))
platforms_world.append((9050, 350, 100, 20))
platforms_world.append((8880, 380, 100, 20))
platforms_world.append((9400, 600 - 150, 40, 15))
platforms_world.append((9520, 600 - 270, 40, 15))
platforms_world.append((9640, 600 - 190, 40, 15))
platforms_world.append((9760, 600 - 250, 40, 15))
platforms_world.append((9880, 600 - 170, 40, 15))
platforms_world.append((10050, 600 - 80, 100, 20))
platforms_world.append((10180, 600 - 200, 100, 20))
platforms_world.append((10310, 600 - 320, 100, 20))
platforms_world.append((10440, 600 - 440, 100, 20))

# Goal
goal_world = (10510, HEIGHT - 470)


# Camera
camera_x = 0
CAMERA_LEFT_MARGIN = WIDTH // 3
CAMERA_RIGHT_MARGIN = WIDTH - WIDTH // 3
WORLD_MIN_X = 0
WORLD_MAX_X = 12000

# Player world position and velocity (world-space)
player_world_x = float(playerStart[0])
player_world_y = float(playerStart[1])
velocity = [0.0, 0.0]
onGround = False

def rect_world_to_screen(rect_world):
    return pygame.Rect(int(rect_world.x - camera_x), int(rect_world.y), rect_world.width, rect_world.height)

def getGameState():
    goal_rect_world = pygame.Rect(goal_world[0], goal_world[1], PLAYER_SIZE[0], PLAYER_SIZE[1])
    player_rect_screen = pygame.Rect(int(player_world_x - camera_x), int(player_world_y), PLAYER_SIZE[0], PLAYER_SIZE[1])
    state = {
        "player": {
            "x": int(player_world_x),
            "y": int(player_world_y),
            "velocityX": velocity[0],
            "velocityY": velocity[1],
            "width": PLAYER_SIZE[0],
            "height": PLAYER_SIZE[1],
            "onGround": onGround
        },
        "goal": {
            "x": goal_rect_world.x,
            "y": goal_rect_world.y,
            "width": PLAYER_SIZE[0],
            "height": PLAYER_SIZE[1]
        },
        "currentLevel": 1,
        "levelComplete": player_rect_screen.colliderect(rect_world_to_screen(goal_rect_world)),
        "platforms": [{"x": x, "y": y, "width": w, "height": h} for (x, y, w, h) in platforms_world],
        "canvasWidth": WIDTH,
        "canvasHeight": HEIGHT,
        "distanceToGoal": ((player_world_x - goal_world[0])**2 + (player_world_y - goal_world[1])**2) ** 0.5,
        "timestamp": time.time()
    }
    return state

def getPossibleActions():
    return ["LEFT", "RIGHT", "JUMP", "IDLE"]

def reset_level():
    global player_world_x, player_world_y, velocity, onGround, camera_x
    player_world_x, player_world_y = float(playerStart[0]), float(playerStart[1])
    velocity = [0.0, 0.0]
    onGround = False
    camera_x = 0

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def update_camera():
    global camera_x
    player_screen_x = player_world_x - camera_x
    if player_screen_x < CAMERA_LEFT_MARGIN:
        camera_x = clamp(int(player_world_x - CAMERA_LEFT_MARGIN), WORLD_MIN_X, WORLD_MAX_X - WIDTH)
    elif player_screen_x > CAMERA_RIGHT_MARGIN:
        camera_x = clamp(int(player_world_x - CAMERA_RIGHT_MARGIN), WORLD_MIN_X, WORLD_MAX_X - WIDTH)

def move_and_collide_world(dx, dy):
    """
    Move player in world-space by dx,dy and resolve collisions against platforms.
    Returns corrected (x,y), vertical on_ground flag.
    """
    x = player_world_x + dx
    y = player_world_y
    # Horizontal move then resolve horizontal collisions (simple)
    player_rect_h = pygame.Rect(int(x), int(y), PLAYER_SIZE[0], PLAYER_SIZE[1])
    for (px, py, pw, ph) in platforms_world:
        plat = pygame.Rect(px, py, pw, ph)
        if player_rect_h.colliderect(plat):
            if dx > 0:
                # hit from left
                x = plat.left - PLAYER_SIZE[0]
                player_rect_h.x = int(x)
            elif dx < 0:
                # hit from right
                x = plat.right
                player_rect_h.x = int(x)

    # Vertical
    y += dy
    player_rect_v = pygame.Rect(int(x), int(y), PLAYER_SIZE[0], PLAYER_SIZE[1])
    on_ground = False
    for (px, py, pw, ph) in platforms_world:
        plat = pygame.Rect(px, py, pw, ph)
        if player_rect_v.colliderect(plat):
            if dy > 0:
                # falling, land on platform
                y = plat.top - PLAYER_SIZE[1]
                velocity[1] = 0.0
                on_ground = True
                player_rect_v.y = int(y)
            elif dy < 0:
                # hitting head
                y = plat.bottom
                velocity[1] = 0.0
                player_rect_v.y = int(y)
    return x, y, on_ground

running = True
show_state = True
reset_level()

while running:
    dt = clock.tick(FPS) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                reset_level()
            if event.key == pygame.K_TAB:
                show_state = not show_state

    keys = pygame.key.get_pressed()

    # Horizontal input (world-space)
    dx = 0.0
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        dx = -move_speed
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        dx = move_speed

    # Gravity
    velocity[1] += gravity
    if velocity[1] > max_fall:
        velocity[1] = max_fall

    # Jump (only if standing on something)
    if (keys[pygame.K_SPACE] or keys[pygame.K_UP] or keys[pygame.K_w]) and onGround:
        velocity[1] = -jump_strength
        onGround = False

    # Apply movement and collisions
    new_x, new_y, grounded = move_and_collide_world(dx, velocity[1])
    player_world_x = clamp(new_x, WORLD_MIN_X, WORLD_MAX_X - PLAYER_SIZE[0])
    player_world_y = new_y
    onGround = grounded

    # If player falls below a threshold, reset
    if player_world_y > HEIGHT + 300:
        reset_level()

    # Camera update
    update_camera()

    # Goal check
    player_rect_screen = pygame.Rect(int(player_world_x - camera_x), int(player_world_y), PLAYER_SIZE[0], PLAYER_SIZE[1])
    goal_rect_world = pygame.Rect(goal_world[0], goal_world[1], PLAYER_SIZE[0], PLAYER_SIZE[1])
    goal_rect_screen = rect_world_to_screen(goal_rect_world)
    level_complete = player_rect_screen.colliderect(goal_rect_screen)

    # Draw
    screen.fill(BG_COLOR)

    # Draw platforms in view
    for (x, y, w, h) in platforms_world:
        plat_world = pygame.Rect(x, y, w, h)
        plat_screen = rect_world_to_screen(plat_world)
        if plat_screen.right >= 0 and plat_screen.left <= WIDTH:
            pygame.draw.rect(screen, PLATFORM_COLOR, plat_screen)

    pygame.draw.rect(screen, GOAL_COLOR, goal_rect_screen)
    pygame.draw.rect(screen, PLAYER_COLOR, player_rect_screen)

    # HUD
    level_text = font.render("Long Level", True, (50, 50, 50))
    screen.blit(level_text, (20, 20))
    coords = font.render(f"Player (world): ({int(player_world_x)}, {int(player_world_y)})", True, (127, 0, 0))
    screen.blit(coords, (20, 50))
    actions = font.render("Controls: Arrow Keys/WASD, Space/Up/W=Jump, R=Restart, TAB=State", True, (0, 0, 127))
    screen.blit(actions, (20, 80))

    if show_state:
        small_font = pygame.font.SysFont(None, 20)
        state = getGameState()
        state_str = json.dumps(state, indent=2)
        lines = state_str.split("\n")[:25]
        for i, l in enumerate(lines):
            s = small_font.render(l[:60], True, (20, 20, 20))
            screen.blit(s, (480, 120 + i * 18))

    if level_complete:
        complete_text = font.render("Goal Reached!", True, (39, 174, 96))
        screen.blit(complete_text, (WIDTH // 2 - 100, HEIGHT // 2))

    pygame.display.flip()

pygame.quit()
sys.exit()
