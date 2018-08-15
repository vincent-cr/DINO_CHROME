import numpy as np
from collections import deque


class Memory():
	def __init__(self, max_size=10000, batch_size=1000):
		self.buffer = deque(maxlen=max_size)
		self.batch_size = batch_size


	def add(self, experience):
		self.buffer.append(experience)


	def add_next_state(self, next_state, episode_counter, is_game_over=False):

		if episode_counter != 0:

			if not is_game_over:
				self.buffer[-2].append(next_state)

			if is_game_over:
				self.buffer[-1].append(next_state)
				self.buffer[-2].append(next_state)

			#print(self.buffer[-2][1], self.buffer[-2][2], self.buffer[-2][3])


	def add_reward(self, reward, is_game_over, episode_counter):

		if episode_counter != 0:

			if not is_game_over:
				self.buffer[-2].append(reward)

			if is_game_over:
				self.buffer[-1].append(reward)
				self.buffer[-2].append(reward)



	def sample(self):
		batch_size = self.batch_size
		buffer_size = len(self.buffer)
		index = np.random.choice(np.arange(buffer_size), size=min(self.batch_size, len(self.buffer)), replace=False)

		return [self.buffer[i] for i in index]
