# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
from transformers import BertTokenizer, BertModel, BertConfig
from collections import namedtuple
import traceback 

import unicodedata
import string
import re
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import utils
import torch.nn.functional as F
from gensim.models import Word2Vec
import pytorch_utils as ptu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 12
FILE_PATH = "data/sighan10.csv"

def loadData():
    df = pd.read_csv(FILE_PATH)
    dataset = df.to_numpy()
    np.random.shuffle(dataset)
    split_index = int(len(dataset)*0.9)
    training_data, test_data = dataset[:split_index, :], dataset[split_index:, :]
    return training_data, test_data

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.next_index = 2  # Count SOS and EOS
        self.correct_confused = None

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.next_index
            self.index2word[self.next_index] = word
            self.next_index += 1 
    
    def addCorrectConfusion(self):
      self.correct_confused = {key: [key] for key in self.index2word.keys()} 
      f = open('confusion.txt',"r")
      for line in f:
        if line[0] in self.word2index.keys():
          correct = self.word2index[line[0]]
          incorrect = line[2:-1]
          for w in incorrect:
            if w in self.word2index.keys():
              self.correct_confused[self.word2index[w]].append(correct)

class Trajectory(object):
    def __init__(self, observations, actions, next_observations, rewards, terminals, length):
        """
            Each argument is a list of what it contains
        """
        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.length = length

class Trainer(object):
    def __init__(self, agent_params, trainer_params):
        self.agent_params = agent_params
        self.n_iter = trainer_params['n_iter']
        self.multiplier = trainer_params['multiplier']
        self.train_n_iter = trainer_params['train_n_iter']
        self.collect_batch_size = trainer_params['batch_size']
        self.train_batch_size = trainer_params['train_batch_size']
        self.env = WeakEnvironment(trainer_params['train'], trainer_params['test'])
        self.reward = []
        self.ep_len = MAX_LENGTH
        self.eval_rewards = []
    def run(self):
        try:
            loaded = torch.load('pg/misc.pt')
            epoch_trained = loaded['epoch']
            reward = loaded['reward']
            eval_rewards = loaded['eval_reward']
            replay_buffer_params = torch.load('pg/replay_buffer.pt')
        except Exception as e:
            print("Exception in loading misc due to", e)
            epoch_trained = 0
            reward = [] 
            eval_rewards = []
            replay_buffer_params = {"paths":[], "obs":None, "action":None, "rewards":None, "next_obs":None, "done":None}

        self.agent_params['replay_buffer'] = replay_buffer_params
        self.agent = PGAgent(self.agent_params, self.env)

        r = 0
        report_period = self.n_iter//100
        try:
            for i in range(self.n_iter):
                paths, envsteps_this_batch, avg_batch_reward = self.collect_training_trajectories()
                r += avg_batch_reward
                self.agent.add_to_replay_buffer(paths)
                self.train()
                if ((i+1)%report_period == 0):
                    # print the reward of the latest 100 steps
                    print("Progress {:.2f}%, with average reward {}".format(i*100/self.n_iter, r/report_period))
                    self.reward.append(r/report_period)
                    r = 0
                    eval_reward = self.evaluate()
                    self.eval_rewards.append(eval_reward)
        except:
            print("Exception has occured, saving models now...")
            #print("Exception due to", e)
            traceback.print_exc()
        finally:
            reward.extend(self.reward) #agglomerate historic rewards
            eval_rewards.extend(self.eval_rewards)
            self.reward = reward
            self.eval_rewards = eval_rewards

            torch.save(self.agent.actor.baseline_decoder.state_dict(), 'pg/baseline_decoder.pt')
            torch.save(self.agent.actor.action_decoder.state_dict(), 'pg/action_decoder.pt')
            torch.save(self.agent.actor.baseline_optimizer.state_dict(), 'pg/baseline_optimizer.pt')
            torch.save({'epoch': epoch_trained + i, 'reward': self.reward, 'eval_reward': self.eval_rewards}, 'pg/misc.pt')
            torch.save({"paths":self.agent.replay_buffer.paths, 
                        "next_obs":self.agent.replay_buffer.next_obs, 
                        "obs":self.agent.replay_buffer.obs, 
                        "action":self.agent.replay_buffer.acs, 
                        "rewards":self.agent.replay_buffer.unconcatenated_rews, 
                        "done":self.agent.replay_buffer.terminals}, 'pg/replay_buffer.pt')
            torch.save(self.agent.env.encoder.state_dict(),'pg/env_encoder.pt')
            torch.save(self.agent.env.decoder.state_dict(),'pg/env_decoder.pt')
            print("Trained {} iterations in total".format(epoch_trained + i))

    def collect_training_trajectories(self):
        timesteps_this_batch = 0
        paths = []
        rewards = 0
        while timesteps_this_batch < self.collect_batch_size:
            path = self.sample_trajectory()
            rewards += sum(path.rewards)
            paths.append(path)
            timesteps_this_batch += path.length

        return paths, timesteps_this_batch, rewards/timesteps_this_batch
    
    def sample_trajectory(self):
        new_ob = self.env.reset()
        obs, acs, rewards, next_obs, terminals = [], [], [], [], []
        steps = 0
        while True:
            obs.append(new_ob)
            ac = self.agent.actor.get_action(new_ob)
            acs.append(ac)
            next_observation, reward, done = self.env.step(new_ob, ac)
            steps += 1
            next_obs.append(next_observation)
            rewards.append(reward)
            rollout_done = 1 if (done or steps >= self.ep_len) else 0
            terminals.append(rollout_done)

            new_ob = next_observation

            if rollout_done:
                break
        return Trajectory(obs, acs, next_obs, rewards, terminals, steps)
    
    def train(self):
        for i in range(self.train_n_iter):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.train_batch_size)
            self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)

    def evaluate(self):
        """print("Training Set Eval")
        for t in self.agent.env.train_data:
            src = t[0]
            trg = t[1]
            self.agent.last_obs = self.agent.env.reset(True, t)
            translated = []
            for i in range(len(trg)):
                action = self.agent.actor.get_actions(self.agent.last_obs)
                obs, reward, done = self.agent.env.step(self.agent.last_obs, action)
                translated.append(self.agent.env.lang.index2word[action.item()])
                self.agent.last_obs = obs
            print('=', src)
            print('<', trg)
            print('>', ''.join(translated))
            print('')
        print("Test Set Eval")"""
        steps, r = 0, 0
        for t in self.env.test_data:
            src = t[0]
            trg = t[1]
            test_obs = self.env.reset(True, t)
            translated = []
            while True:
                ac = self.agent.actor.get_action(test_obs)
                next_observation, reward, done = self.env.step(test_obs, ac)
                steps += 1
                rollout_done = 1 if (done or steps >= self.ep_len) else 0
                test_obs = next_observation

                steps += 1
                r += reward

                if rollout_done:
                    break
        return r/steps

class PGAgent(object):
    def __init__(self, params, env):
        self.replay_buffer = ReplayBuffer(params['replay_buffer'])
        self.env = env
        self.actor = PGPolicy(self.env)
    
    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)
    
    def train(self, observations, actions, rewards_list, next_observations, terminals):
        q_values = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, q_values)
        self.actor.update(observations, actions, advantages, q_values)

    def calculate_q_vals(self, rewards_list):
        q_values = np.concatenate([self.cumsum(r) for r in rewards_list])

        return q_values

    def estimate_advantage(self, obs, q_values):
        baselines_unnormalized = ptu.to_numpy(self.actor.get_baseline(obs))
        baselines = baselines_unnormalized * np.std(q_values) + np.mean(q_values)
        
        advantages = q_values - baselines

        #advantages = q_values.copy()
        return advantages
    
    def cumsum(self, rewards):
        cumsum = np.cumsum(rewards[::-1])[::-1]
        return cumsum

class WeakEnvironment(object):
    def __init__(self, train_data, test_data):
        self.encoder = EncoderRNN()
        try:
          self.encoder.load_state_dict(torch.load('pg/env_encoder.pt'))
        except Exception as e:
          print("Attempting to load env encoder due to", e)
        self.encoder.eval()
        for param in self.encoder.parameters():
          param.requires_grad = False
        self.train_data = train_data
        self.test_data = test_data
        self.input_ids = self.encoder.embed([train_data, test_data]) 
        self.lang = self.encoder.lang
        # decoder doesn't return actions but Q values, so no action distribution, only action based on Q values
        self.decoder = AttnDecoder(self.encoder.hidden_size, self.encoder.input_size)
        try:
          self.decoder.load_state_dict(torch.load('pg/env_decoder.pt'))
        except Exception as e:
          print("Attempting to load env decoder due to", e)
        self.decoder.eval()
        for param in self.decoder.parameters():
          param.requires_grad = False
        if torch.cuda.is_available():
          self.encoder = self.encoder.cuda()
          self.decoder = self.decoder.cuda()
        self.action_space = [i for i in range(self.encoder.input_size)]
        self.criterion = nn.NLLLoss()
        self.env_max_step = MAX_LENGTH
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def random_actions(self):
        #return torch.log_softmax(torch.ones((1, self.encoder.input_size))), np.random.choice(self.action_space)
        action = np.array([np.random.choice(self.action_space)])
        return action
    def step(self, observation, action):
        # observation is [src plain, encoder padded, decoder hidden, curr_input, curr_index]
        # action is the action distribution
        done = False
        target_id = self.input_ids[observation[0]][1]
        curr_index = observation[4]
        
        prev_hidden = torch.from_numpy(observation[2])
        encoder_padded = torch.from_numpy(observation[1])

        action_cur = torch.tensor([[action[0]]]).to(device)
        prev_hidden = prev_hidden.to(device)
        decoded_result = self.decoder(action_cur, prev_hidden, encoder_padded)
        next_hidden = ptu.to_numpy(decoded_result[1].detach())

        # the reward can't be too small, otherwise no signal
        # the reward can't be too large, otherwise will only learn little to be satisified
        # a reward of x means that 1 correct prediction will be killed by x incorrect predictions
        if (action == target_id[curr_index]):
            reward = 10 #5/(((abs(l)**3)+1e-5) + 0.05)
        else:
            reward = -1 #

        if curr_index + 1 == len(target_id):
            done = True
            next_observation = []
        else:
            next_observation = [observation[0], observation[1], next_hidden, action, observation[4]+1]
    
        return next_observation, reward, done
    def reset(self, deterministic_input=False, deterministic_pair=None):
        if deterministic_input:
            pairs = deterministic_pair
        else:
            pairs = random.choice(self.train_data)
        src_plain = pairs[0]

        src_id = self.encoder.input_ids[src_plain][0]
        target_id = self.encoder.input_ids[src_plain][1]
        input_length = len(src_id)
        target_length = len(target_id)

        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = torch.zeros(self.env_max_step, self.encoder.hidden_size, device=device)

        src_id = torch.tensor(src_id).to(device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(src_id[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        encoder_padded = torch.zeros(1, self.env_max_step, self.decoder.hidden_size)
        encoder_padded[:,:len(encoder_outputs),:] = encoder_outputs
        
        decoder_hidden = encoder_hidden

        obs = [None]*5
        obs[0] = src_plain
        obs[1] = ptu.to_numpy(encoder_padded.detach()) # detach here so grad won't propgate to env
        obs[2] = ptu.to_numpy(decoder_hidden.detach()) # detach here so grad won't propgate to env
        obs[3] = np.array([SOS_token])
        obs[4] = 0
        return obs

class PGPolicy(object):

    def __init__(self, env, learning_rate=1e-4):
        self.env = env
        self.lang = self.env.encoder.lang
        self.learning_rate = learning_rate
        self.baseline_decoder = self.build_mlp(self.env.encoder.hidden_size, 1, 3, 32)
        self.action_decoder = AttnDecoder(self.env.encoder.hidden_size, self.env.encoder.input_size)
        try:
          self.baseline_decoder.load_state_dict(torch.load('pg/baseline_decoder.pt'))
          self.baseline_decoder.train()
          self.action_decoder.load_state_dict(torch.load('pg/action_decoder.pt'))
          self.action_decoder.train()
          print("奥利给!Model Loaded!")
        except Exception as e:
          print("Attempting to load policy gradient model but failed due to", e)
        self.baseline_loss = nn.MSELoss()
        self.baseline_optimizer = optim.Adam(
                self.baseline_decoder.parameters(),
                self.learning_rate)
        try: 
          self.baseline_optimizer.load_state_dict(torch.load('pg/baseline_optimizer.pt'))
          print("奥利给!Optimizer Loaded!")
        except Exception as e:
          print("Attempting to load baseline optimizer but failed due to", e)
        

    def build_mlp(self,
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation = nn.Tanh(),
        output_activation = nn.Identity()):

        layers = []
        in_size = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_size, size))
            layers.append(activation)
            in_size = size
        layers.append(nn.Linear(in_size, output_size))
        layers.append(output_activation)
        return nn.Sequential(*layers)

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, ob):
        ob = np.array(ob, dtype=object).reshape(-1,5)
        action_distribution, prob = self.get_action_distribution(ob)
        batch_size = len(ob)

        next_pos_to_predict = ob[:,4].tolist()
        src = ob[:,0].tolist()
        next_id_in_src = [self.lang.word2index[src[i][next_pos_to_predict[i]]] for i in range(batch_size)]
        easily_confused = [self.lang.correct_confused[id] for id in next_id_in_src]

        #action = action_distribution.sample()  # don't bother with rsample
        #taking the most probable action in its respective confusion set
        prob_of_interest = [(easily_confused[i], prob[i,easily_confused[i]].squeeze()) for i in range(batch_size)]
        action = torch.tensor([[prob_of_interest[i][0][torch.argmax(prob_of_interest[i][1])]] for i in range(batch_size)])
        return ptu.to_numpy(action)

    def get_action_distribution(self, ob):
        ob = np.array(ob, dtype=object).reshape(-1,5)
        encoder_padded = ptu.from_numpy(np.array(ob[:,1].tolist()).astype(np.float32))[:,0,:,:]
        decoder_hidden = ptu.from_numpy(np.array(ob[:,2].tolist()).astype(np.float32))[:,0,:,:]
        decoder_input = ptu.from_numpy(np.array(ob[:,3].tolist()).astype(np.float32)).long()

        encoder_padded = encoder_padded.to(device)
        decoder_hidden = decoder_hidden.to(device)
        decoder_input = decoder_input.to(device)

        output, _, _ = self.action_decoder(decoder_input, decoder_hidden, encoder_padded)
        prob = F.softmax(output[:,0,:], dim=1)
        action_distribution = torch.distributions.Categorical(probs=prob)
        return action_distribution, prob
        
    def get_baseline(self, ob):
        ob = np.array(ob, dtype=object).reshape(-1,5)
        decoder_hidden = ptu.from_numpy(np.array(ob[:,2].tolist()).astype(np.float32))[:,0,:,:]

        decoder_hidden = decoder_hidden.to(device)
        value = self.baseline_decoder(decoder_hidden).squeeze()
        return value

    def update(self, observations, actions, advantages, q_values=None):
        observation = np.array(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        
        action_distribution, probs = self.get_action_distribution(observations)
        negative_loglikelihood_predicted = -action_distribution.log_prob(actions)

        advantages = torch.squeeze(advantages)
        loss = torch.dot(negative_loglikelihood_predicted.squeeze(), advantages)
    
        self.action_decoder.optimizer.zero_grad()
        loss.backward()
        self.action_decoder.optimizer.step()

        targets = ptu.normalize(q_values, np.mean(q_values), np.std(q_values))
        targets = ptu.from_numpy(targets)

        baseline_predictions = self.get_baseline(observations)
        baseline_loss = self.baseline_loss(baseline_predictions, targets)

        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()

class ReplayBuffer(object):

    def __init__(self, params, max_size=1000000):

        self.max_size = max_size
        self.paths = params['paths']
        self.obs = params['obs']
        self.acs = params['action']
        self.unconcatenated_rews = params['rewards']
        self.next_obs = params['next_obs']
        self.terminals = params['done']
    
    def convert_listofrollouts(self, paths):
        observations = np.concatenate([path.observations for path in paths])
        actions = np.concatenate([path.actions for path in paths])
        next_observations = np.concatenate([path.next_observations for path in paths])
        terminals = np.concatenate([path.terminals for path in paths])
        unconcatenated_rewards = [path.rewards for path in paths]
        return observations, actions, next_observations, terminals, unconcatenated_rewards

    def add_rollouts(self, paths):
        for path in paths:
            self.paths.append(path)

        observations, actions, next_observations, terminals, unconcatenated_rews = self.convert_listofrollouts(paths)

        if self.obs is None:
            self.obs = observations[-self.max_size:]
            self.acs = actions[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
            self.unconcatenated_rews = unconcatenated_rews[-self.max_size:]
        else:
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]
            if isinstance(unconcatenated_rews, list):
                self.unconcatenated_rews += unconcatenated_rews  # TODO keep only latest max_size around
            else:
                self.unconcatenated_rews.append(unconcatenated_rews)  # TODO keep only latest max_size around

    def sample_recent_data(self, batch_size=1):
        num_recent_rollouts_to_return = 0
        num_datapoints_so_far = 0
        index = -1
        while num_datapoints_so_far < batch_size:
            recent_rollout = self.paths[index]
            index -=1
            num_recent_rollouts_to_return +=1
            num_datapoints_so_far += recent_rollout.length
        rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
        observations, actions, next_observations, terminals, unconcatenated_rews = self.convert_listofrollouts(rollouts_to_return)
        return observations, actions, unconcatenated_rews, next_observations, terminals

class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.lang = None
        self.prepareData()
        self.lang.addCorrectConfusion()
        self.input_size = self.lang.next_index
        self.hidden_size = 256
        self.input_ids = {}
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.optimizer = optim.Adam(self.parameters())
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def prepareData(self):
        print("Reading Chinese Frequency Corpus")
        chinese = Lang("chinese")
        
        df = pd.read_csv(FILE_PATH)
        for s in df['source']:
            chinese.addSentence(s)
        for s in df['reference']:
            chinese.addSentence(s)
        self.lang = chinese

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def indexesFromPair(self, lang, pair):
        input_tensor = self.indexesFromSentence(lang, pair[0])
        target_tensor = self.indexesFromSentence(lang, pair[1])
        return [input_tensor, target_tensor]

    def indexesFromSentence(self, lang, sentence):
    #print(sentence)
        return [lang.word2index[word] for word in sentence]

    def embed(self, datasets):
        for dataset in datasets:
            for d in dataset:
                encodings = self.indexesFromPair(self.lang, d)
                self.input_ids[d[0]] = encodings
        return self.input_ids

class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, learning_rate=0.01, max_length=MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        #self.out = nn.Linear(self.hidden_size, self.output_size)
        self.out = self.build_mlp(self.hidden_size, self.output_size, 3, 32)
        self.optimizer = optim.Adam(self.parameters())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_mlp(self,
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation = nn.Tanh(),
        output_activation = nn.Identity()):

        layers = []
        in_size = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_size, size))
            layers.append(activation)
            in_size = size
        layers.append(nn.Linear(in_size, output_size))
        layers.append(output_activation)
        return nn.Sequential(*layers)
 

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(-1, 1, self.hidden_size)
        
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), 2)), dim=2)
        encoder_outputs = encoder_outputs.to(device)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)

        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)

        output = F.relu(output)
        #print(output.size(), hidden.size())
        output, hidden = self.gru(output.permute(1,0,2), hidden.permute(1,0,2))
        #print(output.size(), hidden.size())

        output = self.out(output)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)