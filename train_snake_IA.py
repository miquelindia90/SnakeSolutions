from src.snake_env import SnakeEnv
from src.dnn import DNN
from src.rl_trainer import RlTrainer

BOARD_SIZE = 200

def main():
    '''Main function.'''
    env = SnakeEnv(board_size=BOARD_SIZE)
    dnn =  DNN(BOARD_SIZE**2, 4, BOARD_SIZE)
    trainer = RlTrainer(env, dnn)
    trainer.train()

if __name__ == '__main__':
    main()