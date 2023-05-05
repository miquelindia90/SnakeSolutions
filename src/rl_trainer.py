import torch

class RlTrainer:
    def __init__(self, env, dnn):#, epochs, batch_size, gamma, epsilon, epsilon_decay, epsilon_min, learning_rate):
        self.env = env
        self.dnn = dnn
        #self.epochs = epochs
        #self.batch_size = batch_size
        #self.gamma = gamma
        #self.epsilon = epsilon
        #self.epsilon_decay = epsilon_decay
        #self.epsilon_min = epsilon_min
        #self.learning_rate = learning_rate
        self._init_optimizer()    
        self._init_loss()

    def _init_optimizer(self):
        pass

    def _init_loss(self):
        pass

    def train(self):

        states = self.env.reset()
        done = False
        while not done:
            states, reward, _, _, done = self.env.step(0)
            print(reward, done)
            self.env.render()
            #env.close()