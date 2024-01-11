import sys
import json

from snake_env import SnakeEnv
from networks.dnn import DNN
from rl_trainer import RlTrainer


def load_json_config(file_name: str) -> dict:
    """Load json file."""
    with open(file_name, "r") as json_file:
        return json.load(json_file)


def main(model_name: str) -> None:
    """Main function."""
    parameters = load_json_config("models/" + model_name + "/config.json")
    env = SnakeEnv(board_size=parameters["board_size"], display=True)
    dnn = DNN(4, parameters["hidden_size"])
    trainer = RlTrainer(env=env, dnn=dnn, parameters=parameters)
    trainer.test(games=25, display=True)


if __name__ == "__main__":
    main(sys.argv[1])
