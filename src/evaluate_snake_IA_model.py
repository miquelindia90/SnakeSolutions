import sys

from snake_env import SnakeEnv
from dnn import DNN
from rl_trainer import RlTrainer

BOARD_SIZE = 150
HIDDEN_SIZE = 100


def main(model_name: str) -> None:
    """Main function."""
    env = SnakeEnv(board_size=BOARD_SIZE, display=False)
    dnn = DNN(4, HIDDEN_SIZE)
    trainer = RlTrainer(env=env, dnn=dnn, model_name=model_name)
    trainer.test(games=1000, display=False)


if __name__ == "__main__":
    main(sys.argv[1])
