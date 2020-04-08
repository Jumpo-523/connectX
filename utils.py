
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from kaggle_environments import evaluate, make
import kaggle_environments
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


class ConnectX(gym.Env):
    def __init__(self, switch_prob=0.5, pair=[]):
        self.env = make('connectx', debug=False)
        self.pair = [None, 'random']
        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob

        # Define required gym fields (examples):
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(config.columns * config.rows)

    def switch_trainer(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)

    def step(self, action):
        return self.trainer.step(action)
    
    def reset(self):
        if np.random.random() < self.switch_prob:
            self.switch_trainer()
        return self.trainer.reset()
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
    
    
class DeepModel(nn.Module):
    def __init__(self, num_states, hidden_units, num_actions):
        super(DeepModel, self).__init__()
        for i in range(len(hidden_units)):
            if i == 0:
                self.hidden_layers.append(
                    nn.Linear(num_states, hidden_units[i])
                )
            else:
                self.hidden_layers.append(
                    nn.Linear(hidden_units[i-1], hidden_units[i])
                )
        self.output_layer = nn.Linear(hidden_units[-1], num_actions)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.sigmoid(layer(x))
        
        # output shape (1, 42)
        x = self.output_layer(x)

        return x

class CNN_model(nn.Module):

    def __init__(self, h, w, outputs):

        super(CNN_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 2, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = CNN_model(6,7, num_actions)# height:6, width: 7 in the game.
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []} # The buffer
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(torch.from_numpy(inputs).float().view(-1, 1, 6, 7))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            # Only start the training process when we have enough experiences in the buffer
            return 0

        # Randomly select n experience in the buffer, n is batch-size
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.preprocess(self.experience['s'][i]) for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])

        # Prepare labels for training process
        states_next = np.asarray([self.preprocess(self.experience['s2'][i]) for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        # import pdb; pdb.set_trace()

        value_next = np.max(TargetNet.predict(states_next).detach().numpy(), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        actions = np.expand_dims(actions, axis=1)
        actions_one_hot = torch.FloatTensor(self.batch_size, self.num_actions).zero_()
        actions_one_hot = actions_one_hot.scatter_(1, torch.LongTensor(actions), 1)
        selected_action_values = torch.sum(self.predict(states) * actions_one_hot, dim=1)
        actual_values = torch.FloatTensor(actual_values)

        self.optimizer.zero_grad()
        loss = self.criterion(selected_action_values, actual_values)
        loss.backward()
        self.optimizer.step()

    # Get an action by using epsilon-greedy
    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return int(np.random.choice([c for c in range(self.num_actions) if state.board[c] == 0]))
        else:
            prediction = self.predict(np.atleast_2d(self.preprocess(state)))[0].detach().numpy()
            for i in range(self.num_actions):
                # もうすでに埋まっているcellは対象外
                if state.board[i] != 0:
                    prediction[i] = -1e7
            return int(np.argmax(prediction))

    # Method used to manage the buffer
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        self.model.load_state_dict(TrainNet.state_dict())

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
    
    # Each state will consist of the board and the mark
    # in the observations
    def preprocess(self, state):
        result = state.board[:]
        # takubo could not understand. 
        # First of all, whose reward was optimized????
        # result.append(state.mark) # comment out by tkb

        return result




def set_direction(d, direction):
    if direction == "row":
        dx, dy = d, 0
    elif direction == "col":
        dx, dy = 0, d
    elif direction == "diag":
        dx, dy = d, d
    elif direction == "anti-diag":
        dx, dy = d, -1*d
    return dx, dy 

def count_seq(new_stone_loc, state,mark):
    """change state for each direction"""
    ans = 0
    i, j = new_stone_loc
    for direction in ["row", "col", "diag", "anti-diag"]:
        count_sequences = 0
        for dir_ in [1, -1]:
            for d in range(4):
                try:
                    dx, dy = set_direction(dir_*d,direction)
                    if dx == 0 and dy == 0:
                        continue
                    elif state[i + dx, j + dy] == mark:
                        count_sequences += 1
                    else:
                        break
                except IndexError:
                    break
        ans = max(count_sequences, ans)
    return ans