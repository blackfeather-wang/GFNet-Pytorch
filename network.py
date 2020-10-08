import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import math

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.hidden = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]


class ActorCritic(nn.Module):
    def __init__(self, feature_dim, state_dim, hidden_state_dim = 1024, policy_conv = True):
        super(ActorCritic, self).__init__()
        
        # encoder with convolution layer for MobileNetV3, EfficientNet and RegNet
        if policy_conv:
            self.state_encoder = nn.Sequential(
                nn.Conv2d(feature_dim, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(int(state_dim*32/feature_dim), hidden_state_dim),
                nn.ReLU()
            )
        # encoder with linear layer for ResNet and DenseNet
        else:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, hidden_state_dim),
                nn.ReLU()
            )

        self.gru = nn.GRU(hidden_state_dim, hidden_state_dim, batch_first=False)
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_state_dim, 2),
            nn.Sigmoid())

        self.critic = nn.Sequential(
            nn.Linear(hidden_state_dim, 1))

        self.hidden_state_dim = hidden_state_dim
        self.policy_conv = policy_conv
        self.feature_dim = feature_dim
        self.feature_ratio = int(math.sqrt(state_dim/feature_dim))

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory, restart_batch=False):
        state = state.view(1, state.size(0), state.size(1))
        if restart_batch:
            del memory.hidden[:]
            memory.hidden.append(torch.zeros(1, state.size(1), self.hidden_state_dim).cuda())

        if self.policy_conv:
            state = self.state_encoder(state[0].view(-1, self.feature_dim, self.feature_ratio, self.feature_ratio))   
        else:
            state = self.state_encoder(state[0])

        state, hidden_output = self.gru(state.view(1, state.size(0), state.size(1)), memory.hidden[-1])
        memory.hidden.append(hidden_output)

        state = state[0]
        action = self.actor(state)
        return action.detach()


class Full_layer(torch.nn.Module):
    def __init__(self, feature_num, hidden_state_dim = 1024, fc_rnn = True, class_num=1000):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.feature_num = feature_num

        self.hidden_state_dim = hidden_state_dim
        self.hidden = None
        self.fc_rnn = fc_rnn
        
        # classifier with RNN for ResNet, DenseNet and RegNet
        if fc_rnn:
            self.rnn = nn.GRU(feature_num, self.hidden_state_dim)
            self.fc = nn.Linear(self.hidden_state_dim, class_num)
        # cascaded classifier for MobileNetV3 and EfficientNet
        else:
            self.fc_2 = nn.Linear(self.feature_num * 2, class_num)
            self.fc_3 = nn.Linear(self.feature_num * 3, class_num)
            self.fc_4 = nn.Linear(self.feature_num * 4, class_num)
            self.fc_5 = nn.Linear(self.feature_num * 5, class_num)

    def forward(self, x, restart=False):

        if self.fc_rnn:
            if restart:
                output, h_n = self.rnn(x.view(1, x.size(0), x.size(1)), torch.zeros(1, x.size(0), self.hidden_state_dim).cuda())
                self.hidden = h_n
            else:
                output, h_n = self.rnn(x.view(1, x.size(0), x.size(1)), self.hidden)
                self.hidden = h_n

            return self.fc(output[0])
        else:
            if restart:
                self.hidden = x
            else:
                self.hidden = torch.cat([self.hidden, x], 1)

            if self.hidden.size(1) == self.feature_num:
                return None
            elif self.hidden.size(1) == self.feature_num * 2:
                return self.fc_2(self.hidden)
            elif self.hidden.size(1) == self.feature_num * 3:
                return self.fc_3(self.hidden)
            elif self.hidden.size(1) == self.feature_num * 4:
                return self.fc_4(self.hidden)
            elif self.hidden.size(1) == self.feature_num * 5:
                return self.fc_5(self.hidden)
            else:
                print(self.hidden.size())
                exit()