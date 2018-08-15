import game
import memory
import dqn


def main():
	mem = memory.Memory(max_size=10000)
	env = game.Game()
	model = dqn.DQN(env)
	env.play_game(mem, model, games=3)


if __name__ == "__main__":
	main()