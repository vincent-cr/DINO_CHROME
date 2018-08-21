import game
import memory
import dqn
import processor


def main():
	mem = memory.Memory()
	img_processor = processor.Processor()
	env = game.Game()
	model = dqn.DQN(env)
	env.play_game(mem, model, img_processor, training_mode=True)


if __name__ == "__main__":
	main()