import numpy as np
from collections import deque
import random


class Memory():
	def __init__(self, max_size=2000, batch_size=32):
		self.buffer = deque(maxlen=max_size)
		self.batch_size = batch_size

		print("Memory created...\n")


	def add(self, experience):
		self.buffer.append(experience)


	def add_next_state(self, next_state, game):

		if game.episode_frames_counter != 0:

			if not game.is_game_over:
				self.buffer[-2].append(next_state)

			if game.is_game_over:
				self.buffer[-1].append(next_state)
				self.buffer[-2].append(next_state)


	def add_reward(self, reward, game):

		if game.episode_frames_counter != 0:

			if not game.is_game_over:
				self.buffer[-2].append(reward)

			if game.is_game_over:
				self.buffer[-1].append(reward)
				self.buffer[-2].append(reward)


	def add_step_count(self, game):

		if game.episode_frames_counter != 0:

			if not game.is_game_over:
				self.buffer[-2].append(game.episode_processed_frames_counter)

			if game.is_game_over:
				self.buffer[-1].append(game.episode_processed_frames_counter)
				self.buffer[-2].append(game.episode_processed_frames_counter+1)


	def sample(self, max_step):
		batch_size = self.batch_size
		buffer_size = len(self.buffer)

		index_low_unsampled = [i for i in range(buffer_size) if self.buffer[i][5] < (max_step/2)]
		index_high_unsampled = [i for i in range(buffer_size) if self.buffer[i][5] >= (max_step/2)]

		index_low_sampled = np.random.choice(np.array(index_low_unsampled), size=min(int(self.batch_size/2), int(buffer_size/2)), replace=True)
		index_high_sampled = np.random.choice(np.array(index_high_unsampled), size=min(int(self.batch_size/2), int(buffer_size/2)), replace=True)

		print(int(buffer_size / 2), self.batch_size / 2, buffer_size, len(index_low_unsampled), len(index_high_unsampled), len(index_low_sampled), len(index_high_sampled), max_step)

		index = [*index_low_sampled, *index_high_sampled]

		return [self.buffer[i] for i in index]
