# Teaching an agent to play Chrome's dino game

## Directories

- game.py: orchestrate the different game elements: environment, memory, network
- memory.py: operations on memory (state, action, is_game_over, reward, next_state)
- dqn.py: networks used to train the agent
- processor.py: captures and process images
- main.py: wraps the game
- chromedriver: used by Selenium for generating the environment
- game_over_black.npy & game_over_white.npy: signatures to identify when game is over
- model: directory to save the model
- checkpoints: directory to save the checkpoints
- Other files are config/log files than can be ignored

To train the agent:

1. Download the files
2. Adapt directories path to match you local env.
3. Run main.py


## Game environment

The program uses Selenium (offline mode) to generate the game. Selenium is a acceptance testing library.
https://selenium-python.readthedocs.io/

## Image preprocessing

The image is captured with CoreGraphics (approx. 30 frames per sec).The region Of Interest (ROI) taken is the 60% left part of the game frame.

The image is then resized to a 80x80 square, filtered (light-grey unharmful obstacles), and transformed into B&W.

![Image processing](https://github.com/vincent-cr/DINO_CHROME/blob/master/resources/image_processing.png)


## Experience replay

In order to train the agent on previous game information, the experiences are stored in memory:

- State (stacked frames)
- Next state
- Action (JUMP or RUN)
- Reward
- is_gameover

![Memory](https://github.com/vincent-cr/DINO_CHROME/blob/master/resources/memory.png)

The frames are stacked as following so the agent can get a notion of motion / acceleration:

![Stacked frames](https://github.com/vincent-cr/DINO_CHROME/blob/master/resources/stacked_images.png)

## DQN

The DQN uses the Bellman Equation:

Bellman Equation

Q(s,a) = r + γ * maxQ(s′,a′)

Where:
s = state (frame)
a = action
r = reward
s’ = next state (stacked frames)
a’ = next action
γ = discount factor

The agent takes the action with the highest predicted Q-value, and is trained on the Loss function as below:

![Q_values](https://github.com/vincent-cr/DINO_CHROME/blob/master/resources/Q_values.png)


The model is improved with Dueling:

![Dueling](https://github.com/vincent-cr/DINO_CHROME/blob/master/resources/Dueling.png)


The final model architecture co:mbines CONV, MAXPOOL, DUELING (FC):

![Model](https://github.com/vincent-cr/DINO_CHROME/blob/master/resources/Model.png)

## Exploration

In order to learn from a variation of SARS element, the game uses ε-Greedy exploration: at each step, exploration rate is decreased. However, the game is highly sensitive to wrong explorations (game over).
