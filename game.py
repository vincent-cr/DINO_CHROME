from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import Quartz.CoreGraphics as CG
import numpy as np
from skimage import transform
import time
from datetime import datetime
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


class Game():

    def __init__(self):
        self.driver = webdriver.Chrome(executable_path="/Users/Vincent/PycharmProjects/IAM_dino/v2/chromedriver")
        self.driver.set_network_conditions(offline=True, latency=1, throughput=100)
        self.driver.get('http://www.python.org/')
        self.driver.set_window_rect(x=0, y=0, width=640, height=320)
        self.element = self.driver.find_element_by_id("main-content")
        self.actions = ActionChains(self.driver)
        self.actions.send_keys(Keys.ARROW_UP)

        self.state_size = [80, 80, 4]
        self.action_size = 2
        self.possible_actions = np.array(np.identity(self.action_size,dtype=int).tolist())
        self.exploration_rate = 1.
        self.explore_decay = 0.997
        self.min_exploration_rate = 0.02
        self.skip_frames = 28
        self.episodes = 1

        self.is_game_over = False
        self.episode_frames_counter = -1
        self.step = 0
        self.game_count = 1

        print("Environment created...\n")

        # Take a screenshot to make sure the window is on the front
        self.driver.get_screenshot_as_png()


    def reset_game(self):
        self.seq_frames = deque(maxlen=4)
        self.episode_frames_counter = -1
        self.is_game_over = False
        self.game_count += 1


    def get_reward(self, memory):

        if self.episode_frames_counter != 0:

            # Get previous action from memory
            prev_action = np.argmax(memory.buffer[-1][1])

            # If game over
            if self.is_game_over:
                reward = -100
            # If jump
            elif prev_action == 1:
                reward = -3
            # If not jump
            elif prev_action == 0:
                reward = 1

        else:
            reward = 0

        return reward


    def perform_action(self, predicted_action):

        # Pick random number
        is_exploration = np.random.random_sample()

        # Only perform action if not game over
        if not self.is_game_over:

            # If explore
            if is_exploration < max(self.exploration_rate, self.min_exploration_rate):
                # 50% chance to jump, 50% chance to run
                if np.random.random_sample() > 0.5:
                    self.actions.perform()
                    action = 1
                    action_info = "EXPLORATION \t JUMP\t"
                else:
                    action = 0
                    action_info = "EXPLORATION \t RUN\t"
                action_array = self.possible_actions[action]

            # If not explore
            if is_exploration >= max(self.exploration_rate, self.min_exploration_rate):
                if predicted_action == 1:
                    action = 1
                    self.actions.perform()
                    action_info = "ACTION \t\t\t JUMP\t"
                else:
                    action = 0
                    action_info = "ACTION \t\t\t RUN\t"

                action_array = self.possible_actions[action]

        # If game over, no action
        if self.is_game_over:
            action = 0
            action_array = self.possible_actions[action]
            action_info = "NO ACTION\t\t\t\t"

        self.exploration_rate *= self.explore_decay

        return action, action_array, action_info


    def play_game(self, memory, model, processor, training_mode=True):

        # Initialize tf session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.logging.set_verbosity(tf.logging.INFO)

            # Save model
            tf.saved_model.simple_save(
                    sess,
                    inputs={"x":model.inputs_, "y":model.target_Q},
                    outputs={"z":model.Q},
                    export_dir=model.dir_saved_model,
                    legacy_init_op=None)

            # Play all games
            while True:
                for i in range(self.episodes):
                    # Start game
                    self.actions.perform()
                    print("START GAME {}".format(self.game_count))
                    time.sleep(0.5)
                    t_start = time.time()

                    while True:

                        # Capture raw frame
                        img = processor.img_grab()

                        # Update the frame counter (for this episode)
                        self.episode_frames_counter += 1

                        # Only process and stack each 4th frame
                        if self.episode_frames_counter % self.skip_frames == 0:

                            # Process image & update is_game_over
                            img_processed = processor.img_process(img, self)

                            # Stack images
                            state = processor.img_stack(img_processed, self)

                            # Predict action
                            predicted_action, Q = model.predict_action(state, sess)

                            # Perform action - may be exploration
                            action, action_array, action_info = self.perform_action(predicted_action)

                            # Calculate reward
                            reward = self.get_reward(memory)

                            # Add current experience to memory
                            memory.add([state, action_array, self.is_game_over])

                            # Add reward to previous experience
                            memory.add_reward(reward, self.is_game_over, self.episode_frames_counter)

                            # Add state (stacked frames) as next state of previous state
                            memory.add_next_state(state, self.episode_frames_counter, self.is_game_over)


                            # Update the step counter
                            self.step += 1

                            # Print step summary
                            print(action_info, "Game over: {}".format(self.is_game_over), "\t\t", round(Q[0][0], 2), "\t",  round(Q[0][1], 2), "\t\t reward: {}".format(reward))

                            # If gameover
                            if self.is_game_over:

                                t_end = time.time()
                                t_game = round((t_end - t_start) % 60, 2)

                                print("GAME OVER", "-", "Game {}".format(self.game_count), "-", "Game duration: {} s.\n".format(t_game), "\t Memory size: {}".format(len(memory.buffer)))

                                # Reset game
                                self.reset_game()
                                time.sleep(0.5)
                                break

                # Train
                print("\n... TRAING MODEL ... \n")
                if self.game_count <= 80:
                    steps = 20
                else:
                    steps = 10

                for i in range(steps):
                    if training_mode:
                        model.train(memory, sess)
                time.sleep(0.5)

                # Save checkpoints every 100 games
                if self.game_count % 100 == 0:
                    print("\n... SAVING MODEL ... \n")
                    model.save(sess, self.game_count)
                    time.sleep(4)

                    #self.exploration_rate = 0.6


