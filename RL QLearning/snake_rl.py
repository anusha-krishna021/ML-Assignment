import pygame
import random
import numpy as np
import matplotlib.pyplot as plt

# --------------------
# SETTINGS
# --------------------
GRID_SIZE = 20
WIDTH = 400
HEIGHT = 400

EPISODES = 300

alpha = 0.1
gamma = 0.9
epsilon = 0.2

# Directions
UP = (0, -GRID_SIZE)
DOWN = (0, GRID_SIZE)
LEFT = (-GRID_SIZE, 0)
RIGHT = (GRID_SIZE, 0)

directions = [UP, RIGHT, DOWN, LEFT]

Q = {}

# --------------------
# Q FUNCTIONS
# --------------------


def get_q(state):
    if state not in Q:
        Q[state] = [0, 0, 0]
    return Q[state]


def choose_action(state):
    if random.random() < epsilon:
        return random.randint(0, 2)
    return np.argmax(get_q(state))


def update_q(state, action, reward, next_state):
    q_values = get_q(state)
    next_max = max(get_q(next_state))

    q_values[action] += alpha * (
        reward + gamma * next_max - q_values[action]
    )

# --------------------
# GAME FUNCTIONS
# --------------------


def is_collision(point, snake):
    x, y = point
    if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
        return True
    if point in snake[1:]:
        return True
    return False


def move(snake, direction):
    head = snake[0]
    new_head = (head[0] + direction[0], head[1] + direction[1])
    snake.insert(0, new_head)
    return snake


def update_direction(direction, action):
    idx = directions.index(direction)

    if action == 0:
        return directions[idx]
    elif action == 1:
        return directions[(idx + 1) % 4]
    else:
        return directions[(idx - 1) % 4]


def get_state(snake, food, direction):
    head = snake[0]

    def danger(dir):
        next_point = (head[0] + dir[0], head[1] + dir[1])
        return int(is_collision(next_point, snake))

    left_dir = directions[(directions.index(direction) - 1) % 4]
    right_dir = directions[(directions.index(direction) + 1) % 4]

    return (
        danger(direction),
        danger(left_dir),
        danger(right_dir),
        int(food[0] < head[0]),
        int(food[0] > head[0]),
        int(food[1] < head[1]),
        int(food[1] > head[1])
    )

# --------------------
# DRAW FUNCTION
# --------------------


def draw(win, snake, food, score, direction):
    win.fill((0, 0, 0))

    for i, s in enumerate(snake):
        if i == 0:
            pygame.draw.rect(win, (0, 200, 0), (*s, GRID_SIZE, GRID_SIZE))

            # Direction-based eyes 👀
            if direction == RIGHT:
                eyes = [(s[0]+15, s[1]+5), (s[0]+15, s[1]+15)]
            elif direction == LEFT:
                eyes = [(s[0]+5, s[1]+5), (s[0]+5, s[1]+15)]
            elif direction == UP:
                eyes = [(s[0]+5, s[1]+5), (s[0]+15, s[1]+5)]
            else:  # DOWN
                eyes = [(s[0]+5, s[1]+15), (s[0]+15, s[1]+15)]

            for eye in eyes:
                pygame.draw.circle(win, (255, 255, 255), eye, 3)
        else:
            pygame.draw.rect(win, (0, 255, 0), (*s, GRID_SIZE, GRID_SIZE))

    pygame.draw.rect(win, (255, 0, 0), (*food, GRID_SIZE, GRID_SIZE))

    font = pygame.font.SysFont(None, 30)
    text = font.render(f"Score: {score}", True, (255, 255, 255))
    win.blit(text, (10, 10))

    pygame.display.update()


# --------------------
# MAIN TRAINING
# --------------------
pygame.init()
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake AI")
clock = pygame.time.Clock()

scores = []

for ep in range(EPISODES):

    snake = [(200, 200)]
    direction = RIGHT
    food = (random.randrange(0, WIDTH, GRID_SIZE),
            random.randrange(0, HEIGHT, GRID_SIZE))

    score = 0
    steps = 0
    done = False

    while not done and steps < 200:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        state = get_state(snake, food, direction)
        action = choose_action(state)
        direction = update_direction(direction, action)

        snake = move(snake, direction)
        head = snake[0]

        if is_collision(head, snake):
            reward = -10
            done = True

        elif head == food:
            reward = 10
            score += 1
            food = (random.randrange(0, WIDTH, GRID_SIZE),
                    random.randrange(0, HEIGHT, GRID_SIZE))

        else:
            reward = -0.1
            snake.pop()

        next_state = get_state(snake, food, direction)
        update_q(state, action, reward, next_state)

        draw(win, snake, food, score, direction)

        clock.tick(8)
        steps += 1

    scores.append(score)
    print(f"Episode {ep+1}, Score: {score}")

    # Game over pause
    pygame.time.delay(200)

pygame.quit()

# --------------------
# GRAPH
# --------------------
plt.plot(scores)
plt.xlabel("Episodes")
plt.ylabel("Score")
plt.title("Learning Progress")
plt.show()
