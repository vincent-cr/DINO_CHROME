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
        self.driver.set_network_conditions(offline=True, latency=5, throughput=100)
        self.driver.get('http://www.python.org/')
        self.driver.set_window_rect(x=0, y=0, width=640, height=320)

        self.element = self.driver.find_element_by_id("main-content")

        self.actions = ActionChains(self.driver)
        self.actions.send_keys(Keys.ARROW_UP)
        self.state_size = [80, 80, 4]
        self.action_size = 2
        self.possible_actions = np.array(np.identity(self.action_size,dtype=int).tolist())
        self.exploration_rate = 0.30
        self.min_exploration_rate = 0.02
        self.skip_frames = 16

        game_over_eval_black = np.load("game_over_black.npy")
        game_over_eval_white = np.load("game_over_white.npy")
        self.game_over_state = [game_over_eval_black, game_over_eval_white]

        self.is_game_over = False
        self.seq_frames = deque(maxlen=4)
        self.episode_frames_counter = -1
        self.step = 0
        self.game_count = 1

        # ROI 75% horizontal
        self.roi = CG.CGRectMake(10, 140, 640*0.75, 140)

        # Take a screenshot to make sure the window is on the front
        self.driver.get_screenshot_as_png()


    def grab_img(self):

        img = CG.CGWindowListCreateImage(
                self.roi,
                CG.kCGWindowListOptionOnScreenOnly,
                CG.kCGNullWindowID,
                CG.kCGWindowImageDefault)

        pixeldata = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(img))
        width = CG.CGImageGetWidth(img)
        height = CG.CGImageGetHeight(img)
        bytesperrow = CG.CGImageGetBytesPerRow(img)
        img_mat = np.frombuffer(pixeldata, dtype=np.uint8).reshape((height, bytesperrow // 4, 4))

        return img_mat[:, :width, :]


    def eval_game_over(self, img):
        for condition in self.game_over_state:
            if np.array_equal(img[168:225, 590:647], condition):
                self.is_game_over = True



    def reset_game(self):
        self.seq_frames = deque(maxlen=4)
        self.episode_frames_counter = -1
        self.is_game_over = False
        self.game_count += 1


    def img_process(self, img):
        # From  (row, col, ch) to (row, col)
        img_grey = img.mean(axis=-1, keepdims=0)

        # Evaluate if game is over
        self.eval_game_over(img_grey)

        # Downsample image
        img_resized = transform.resize(img_grey, [80, 80])

        # Convert to B&W
        img_bw = np.where(img_resized > np.mean(img_resized), 1, 0)

        return img_bw


    def stack_img(self, img):

        # If first frame, we stack the deque with 4 identical frames
        if self.episode_frames_counter == 0:
            self.seq_frames.append(img)
            self.seq_frames.append(img)
            self.seq_frames.append(img)
            self.seq_frames.append(img)

        # Else, we append only the current frame
        else:
            self.seq_frames.append(img)

        # We stack the 4 consecutive frames into a 80x80x4 matrice
        stacked_frames = np.stack(self.seq_frames, axis=2)

        return stacked_frames


    def get_reward(self, memory):

        if self.episode_frames_counter != 0:

            # Get previous action from memory
            prev_action = np.argmax(memory.buffer[-1][1])

            # If game over
            if self.is_game_over:
                reward = -100
            # If jump
            elif prev_action == 1:
                reward = -10
            # If not jump
            elif prev_action == 0:
                reward = 10

        else:
            reward = 0

        return reward


    def perform_action(self, predicted_action):

        # Pick random number
        is_exploration = np.random.random_sample()

        # Only perform action if not game over
        if not self.is_game_over:

            # If explore
            if is_exploration < self.exploration_rate:
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
            if is_exploration >= self.exploration_rate:
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

        if self.exploration_rate >= self.min_exploration_rate:
            self.exploration_rate -= 0.000001

        return action, action_array, action_info


    def play_game(self, memory, model, games=7, training_mode=True):

        # Initialize tf session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.logging.set_verbosity(tf.logging.INFO)

            # Play all games
            while True:
                for i in range(games):
                    # Start game
                    self.actions.perform()
                    print("START GAME {}".format(self.game_count))
                    time.sleep(0.5)
                    t_start = datetime.now()

                    while True:

                        # Capture raw frame
                        img = self.grab_img()

                        # Update the frame counter (for this episode)
                        self.episode_frames_counter += 1

                        # Only process and stack each 4th frame
                        if self.episode_frames_counter % self.skip_frames == 0:

                            # Process image & update is_game_over
                            img_processed = self.img_process(img)

                            # Stack images
                            state = self.stack_img(img_processed)

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
                            print(action_info, "Game over: {}".format(self.is_game_over), "\t", round(Q[0][0], 2), "\t",  round(Q[0][1], 2))

                            # If gameover
                            if self.is_game_over:

                                t_end = datetime.now()
                                t_game = t_end - t_start

                                print("GAME OVER", "-", "Game {}".format(self.game_count), "-", "Game duration: {}\n".format(t_game))

                                # Reset game
                                self.reset_game()
                                time.sleep(2)
                                break

                # Train
                print("\n... TRAING MODEL ... \n")
                if training_mode:
                    model.train(memory, sess)
