import sys
import argparse

from snake_env import SnakeEnv
from dnn import DNN
from rl_trainer import RlTrainer



def main(parameters: argparse.Namespace) -> None:
    """Main function."""
    env = SnakeEnv(board_size=parameters["board_size"])
    dnn = DNN(4, parameters["hidden_size"])
    trainer = RlTrainer(env=env, dnn=dnn, parameters=parameters)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="Name of the model to be trained")
    parser.add_argument("--board_size", type=int, default=150, help="Size of the board")

    # DNN Configuration
    parser.add_argument("--hidden_size", type=int, default=100, help="Size of the hidden layer")

    # Training Configuration
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--max_episodes", type=int, default=1_000_000, help="Maximum number of episodes")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--initial_epsilon", type=float, default=0.5, help="Initial epsilon value")
    parser.add_argument("--target_epsilon", type=float, default=0.01, help="Maximum epsilon value")
    parser.add_argument("--epsilon_converge_ratio", type=float, default=1 / 3, help="Ratio for epsilon convergence")
    parser.add_argument("--buffer_size", type=int, default=5000, help="Size of the replay buffer")

    parameters = vars(parser.parse_args())
    main(parameters)
