import pygame
import random
import sys
import numpy as np

#The training takes a 5 seconds please be patient 

BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
WINDOW_HEIGHT = 400
WINDOW_WIDTH = 400

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

NUM_RAND_BLOCKS = 100
rand_generated_blocks = None

def main():
    global SCREEN, CLOCK
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BLACK)

    # Initialize Q-values
    q_values = np.zeros((WINDOW_WIDTH // 20, WINDOW_HEIGHT // 20, 4))

    # Set the exit position and obstacles
    exit_pos = (WINDOW_WIDTH // 20 - 1, WINDOW_HEIGHT // 20 - 1)
    obstacles = [(b[0] // 20, b[1] // 20) for b in rand_generated_blocks]

    # Train the agent (Q-learning)
    train(q_values, exit_pos, obstacles)
    while True:
        # Test the trained agent
        test(q_values, exit_pos, obstacles)
        #main()
def train(q_values, exit_pos, obstacles):
    num_episodes = 5000
    learning_rate = 0.1
    discount_factor = 0.99
    exploration_prob = 1.0
    min_exploration_prob = 0.01
    exploration_decay = 0.995

    for episode in range(num_episodes):
        # Initialize the starting position of the agent
        agent_pos = (0, 0)

        while agent_pos != exit_pos:
            # Choose an action (exploration vs. exploitation)
            if random.uniform(0, 1) < exploration_prob:
                action = random.randint(0, 3)  # Random action
            else:
                action = np.argmax(q_values[agent_pos[0], agent_pos[1]])

            # Perform the action and observe the reward and new state
            new_agent_pos = move(agent_pos, action)

            # Check if the new position is within the maze bounds
            if (
                0 <= new_agent_pos[0] < WINDOW_WIDTH // 20
                and 0 <= new_agent_pos[1] < WINDOW_HEIGHT // 20
            ):
                reward = get_reward(new_agent_pos, exit_pos, obstacles)

                # Update Q-value using Q-learning update rule
                if new_agent_pos == exit_pos:
                    q_values[agent_pos[0], agent_pos[1], action] = reward
                else:
                    q_values[agent_pos[0], agent_pos[1], action] = (
                        1 - learning_rate
                    ) * q_values[agent_pos[0], agent_pos[1], action] + learning_rate * (
                        reward + discount_factor * np.max(q_values[new_agent_pos[0], new_agent_pos[1]])
                    )

                agent_pos = new_agent_pos

        # Decay exploration probability
        exploration_prob = max(exploration_prob * exploration_decay, min_exploration_prob)



def test(q_values, exit_pos, obstacles):
    # Initialize the starting position of the agent
    agent_pos = (0, 0)

    while agent_pos != exit_pos:
        # Choose the best action
        action = np.argmax(q_values[agent_pos[0], agent_pos[1]])

        # Perform the action and observe the new state
        new_agent_pos = move(agent_pos, action)

        # Update agent position
        agent_pos = new_agent_pos

        # Draw the maze and agent
        drawGrid()
        draw_agent(agent_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()
        CLOCK.tick(5)
        #global rand_generated_blocks
        #rand_generated_blocks = random_block()

def move(pos, action):
    # Move the agent based on the chosen action
    if action == 0:  # Move up
        return pos[0] - 1, pos[1]
    elif action == 1:  # Move down
        return pos[0] + 1, pos[1]
    elif action == 2:  # Move left
        return pos[0], pos[1] - 1
    elif action == 3:  # Move right
        return pos[0], pos[1] + 1

def get_reward(agent_pos, exit_pos, obstacles):
    # Check if the agent position is an obstacle or the exit
    if agent_pos in obstacles:
        return -100  # Negative reward for hitting an obstacle
    elif agent_pos == exit_pos:
        return 100  # Positive reward for reaching the exit
    else:
        return 0  # No reward for normal moves

def random_block() -> []:
    random_blocks= []
    for i in range(NUM_RAND_BLOCKS):
        rand_x = random.randrange(0,36, 2)
        rand_y = random.randrange(0,36,2)
        rand_x *= 10
        rand_y *=10
        random_blocks.append([rand_x,rand_y])
    return random_blocks

def draw_blocks(blockSize):
    pygame.draw.rect(SCREEN, GREEN, ((blockSize * blockSize) - 20, (blockSize * blockSize) - 20, 20, 20))  # exit
    for i in rand_generated_blocks:
        pygame.draw.rect(SCREEN, BLUE, (i[0], i[1], 20, 20))

def drawGrid():
    blockSize = 20  # Set the size of the grid block
    for x in range(0, WINDOW_WIDTH, blockSize):
        for y in range(0, WINDOW_HEIGHT, blockSize):
            rect = pygame.Rect(x, y, blockSize, blockSize)
            pygame.draw.rect(SCREEN, WHITE, rect, 1)

    draw_blocks(20)

def draw_agent(agent_pos):
    blockSize = 20
    agent_rect = pygame.Rect(agent_pos[0] * blockSize, agent_pos[1] * blockSize, blockSize, blockSize)
    pygame.draw.rect(SCREEN, RED, agent_rect)

if __name__ == "__main__":
    rand_generated_blocks = random_block()
    main()
