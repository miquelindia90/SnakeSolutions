import os
import json
import argparse

from snake_env import SnakeEnv
from networks.dnn import DNN
from networks.cnn import CNN
from rl_trainer import RlTrainer


def _init_model_folders(model_name: str) -> None:
    """Initialize the model folder."""
    if not os.path.exists("models/{}".format(model_name)):
        os.mkdir("models/{}".format(model_name))


def _save_config_data(model_name: str, parameters: dict) -> None:
    with open("models/" + model_name + "/config.json", "w") as json_handler:
        json.dump(parameters, json_handler, indent=4)


def main(parameters: dict) -> None:
    """Main function."""
    _init_model_folders(parameters["model_name"])
    _save_config_data(parameters["model_name"], parameters)
    env = SnakeEnv(board_size=parameters["board_size"])
    if parameters["network"] == "DNN":
        dnn = DNN(4, parameters["hidden_size"])
    elif parameters["network"] == "CNN":
        dnn = CNN(env.get_surface_dimensions(), 4, parameters["hidden_size"])
    trainer = RlTrainer(env=env, dnn=dnn, parameters=parameters)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        "-n",
        type=str,
        required=True,
        help="Name of the model to be trained",
    )
    parser.add_argument("--board_size", type=int, default=150, help="Size of the board")

    # DNN Configuration
    parser.add_argument(
        "--network",
        type=str,
        default="CNN",
        choices=["DNN", "CNN"],
        help="Size of the hidden layer",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=100, help="Size of the hidden layer"
    )

    # Training Configuration
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "--max_episodes", type=int, default=1_000_000, help="Maximum number of episodes"
    )
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument(
        "--initial_epsilon", type=float, default=0.5, help="Initial epsilon value"
    )
    parser.add_argument(
        "--target_epsilon", type=float, default=0.01, help="Maximum epsilon value"
    )
    parser.add_argument(
        "--epsilon_converge_ratio",
        type=float,
        default=1 / 3,
        help="Ratio for epsilon convergence",
    )
    parser.add_argument(
        "--buffer_size", type=int, default=5000, help="Size of the replay buffer"
    )

    parameters = vars(parser.parse_args())
    main(parameters)
