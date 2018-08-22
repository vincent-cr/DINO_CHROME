import Quartz.CoreGraphics as CG
import numpy as np
from skimage import transform
import time
from datetime import datetime
from collections import deque
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


class Processor():
	def __init__(self):

		# ROI 75% horizontal
		self.roi_horizontal = 0.55
		self.roi = CG.CGRectMake(10, 140, 640 * self.roi_horizontal, 140)
		self.seq_frames = deque(maxlen=4)

		game_over_pix_black = np.load("game_over_black.npy")
		game_over_pix_white = np.load("game_over_white.npy")
		self.game_over_state = [game_over_pix_black, game_over_pix_white]

		print("Image processor created...\n")


	def img_grab(self):
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


	def img_process(self, img, game):
		# From  (row, col, ch) to (row, col)
		img_grey = img.mean(axis=-1, keepdims=0)
		img_copy = img_grey

		# Evaluate if game is over
		self.eval_game_over(img_grey, game)

		# Filter out unharmfull obstacles ( light grey)
		img_grey[img_copy > 200] = 255.
		img_grey[img_copy < 200] = 0.

		# Downsample image
		img_resized = transform.resize(img_grey, [80, 80])

		# Convert to B&W, where black gets higher values (for maxpooling)
		img_bw = np.where(img_resized > 200, 0., 1.)

		# plt.imshow(img_bw, cmap=plt.cm.Greys_r)
		# plt.show()
		return img_bw


	def img_stack(self, img, game):
		# If first frame, we stack the deque with 4 identical frames
		if game.episode_frames_counter == 0:
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


	def eval_game_over(self, img, game):
		for condition in self.game_over_state:
			if np.array_equal(img[168:225, 590:647], condition):
				game.is_game_over = True
