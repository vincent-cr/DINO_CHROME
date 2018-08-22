import tensorflow as tf
import numpy as np
from datetime import datetime

class DQN:
	def __init__(self, game, name='DQN'):
		self.learning_rate = 0.0002
		self.state_size = game.state_size
		self.action_size = game.action_size
		self.possible_actions = game.possible_actions
		self.gamma = 0.95
		self.dir_saved_checkpoints = "/Users/Vincent/PycharmProjects/IAM_dino/v2/checkpoints/"
		self.dir_saved_model = "/Users/Vincent/PycharmProjects/IAM_dino/v2/model/" + str(datetime.now())

		with tf.variable_scope(name):

			# Create placeholders
			self.inputs_ = tf.placeholder(tf.float32, [None, *self.state_size], name="inputs")
			self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

			# target_Q is the R(s,a) + ymax Qhat(s', a')
			self.target_Q = tf.placeholder(tf.float32, [None], name="target")


			# CONV1
			self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
										  filters=32,
										  kernel_size=[8, 8],
										  strides=[4, 4],
										  padding="VALID",
										  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
										  name="conv1")

			self.conv1_out = tf.nn.relu(self.conv1, name="conv1_out")


			# MAX POOL
			self.max_pool1 =  tf.nn.max_pool(self.conv1_out, [1, 4, 4, 1], [1, 2, 2, 1], padding='SAME', name="max_pool")

			# CONV2
			self.conv2 = tf.layers.conv2d(inputs=self.max_pool1,
										  filters=64,
										  kernel_size=[4, 4],
										  strides=[2, 2],
										  padding="VALID",
										  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
										  name="conv2")

			self.conv2_out = tf.nn.relu(self.conv2, name="conv2_out")


			# CONV3
			self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
										  filters=64,
										  kernel_size=[3, 3],
										  strides=[1, 1],
										  padding="VALID",
										  kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
										  name="conv3")

			self.conv3_out = tf.nn.relu(self.conv3, name="conv3_out")


			self.flatten = tf.contrib.layers.flatten(self.conv3_out)

			self.fc = tf.layers.dense(inputs=self.flatten,
									  units=512,
									  activation=tf.nn.relu,
									  kernel_initializer=tf.contrib.layers.xavier_initializer(),
									  name="fc1")

			self.output = tf.layers.dense(inputs=self.fc,
										  kernel_initializer=tf.contrib.layers.xavier_initializer(),
										  units=self.action_size,
										  activation=None)

			# Q is our predicted Q value.
			self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

			# The loss is the difference between our predicted Q_values and the Q_target Sum(Qtarget - Q)^2
			self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

			self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

			# Init saver
			self.saver = tf.train.Saver()

			print("Model created...\n")


	def predict_action(self, state, sess):

		# Estimate the Qs values state
		Qs = sess.run(self.output, feed_dict={self.inputs_: state.reshape((1, *state.shape))})

		# Take the biggest Q value (= the best action)
		action = np.argmax(Qs)

		return action, Qs


	def train(self, memory, sess):
		batch = memory.sample()
		states_mb = np.array([each[0] for each in batch], ndmin=3)
		actions_mb = np.array([each[1] for each in batch])
		is_game_over_mb = np.array([each[2] for each in batch])
		rewards_mb = np.array([each[3] for each in batch])
		next_states_mb = np.array([each[4] for each in batch], ndmin=3)

		target_Qs_batch = []


		# Get Q values for next_state
		Qs_next_state = sess.run(self.output, feed_dict={self.inputs_: next_states_mb})

		# Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
		for i in range(0, len(batch)):
			is_game_over = is_game_over_mb[i]

			# If we are in a terminal state, only equals reward
			if is_game_over:
				target_Qs_batch.append(rewards_mb[i])

			else:
				target = rewards_mb[i] + self.gamma * np.max(Qs_next_state[i])
				target_Qs_batch.append(target)

		targets_mb = np.array([each for each in target_Qs_batch])

		loss, _ = sess.run([self.loss, self.optimizer],
						   feed_dict={self.inputs_: states_mb,
									  self.target_Q: targets_mb,
									  self.actions_: actions_mb})
		print(loss)


	def save(self, session, count):
		self.saver.save(session, self.dir_saved_checkpoints + "dino.ckpt", global_step=count)
		print("...Model saved...")

	def load(self, session, checkpoint_name):
		self.saver.restore(session, checkpoint_name)
		print("Model loaded:", checkpoint_name)




