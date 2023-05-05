from src.snake_env import SnakeEnv

def main():

    env = SnakeEnv()
    states = env.reset()
    done = False
    while not done:
        #states, reward, _, _, done = env.step(env.action_space.sample())
        states, reward, _, _, done = env.step(0)
        print(reward, done)
        env.render()

if __name__ == '__main__':
    main()