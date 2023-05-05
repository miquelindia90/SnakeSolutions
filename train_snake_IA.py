from src.snake_env import SnakeEnv
from src.dnn import DNN
from src.rl_trainer import RlTrainer


def main():

    env = SnakeEnv()
    dnn =  DNN(24, 4, 24)
    trainer = RlTrainer(env, dnn)
    trainer.train()

if __name__ == '__main__':
    main()