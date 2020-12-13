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

class Trainer(object):
    def __init__(self, agent_params, trainer_params):
        self.agent_params = agent_params
        self.n_iter = trainer_params['n_iter']
        self.multiplier = trainer_params['multiplier']
        self.train_n_iter = trainer_params['train_n_iter']
        self.train_batch_size = trainer_params['train_batch_size']
        self.reward = []
        #self.eval_rewards = []
        self.precision = []
        self.recall = []
        self.fhalf = []

    def run(self):
        try:
            loaded = torch.load('q_test/misc.pt')
            epoch_trained = loaded['epoch']
            reward = loaded['reward']
            #eval_rewards = loaded['eval_reward']
            precision = loaded['precision']
            recall = loaded['recall']
            fhalf = loaded['fhalf']
            t = loaded['t']
            num_param_updates = loaded['num_param_updates']
            replay_buffer_params = torch.load('q_test/replay_buffer.pt')
        except Exception as e:
            print("Exception in loading misc due to", e)
            epoch_trained = 0
            reward = [] 
            #eval_rewards = []
            precision = []
            recall = []
            fhalf = []
            t = 0
            num_param_updates = 0
            replay_buffer_params = {"next_idx":0, "num_in_buffer":0, "obs":None, "action":None, "reward":None, "done":None}

        self.agent_params['t'] = t
        self.agent_params['num_param_updates'] = num_param_updates
        self.agent_params['replay_buffer'] = replay_buffer_params
        self.agent = DQNAgent(self.agent_params)

        r = 0
        report_period = self.n_iter//100
        try:
            for i in range(self.n_iter):
                r += self.agent.step()
                self.train()
                if ((i+1)%report_period == 0):
                    # print the reward of the latest 100 steps
                    print("Progress {:.2f}%, with average reward {}".format(i*100/self.n_iter, r/report_period))
                    self.reward.append(r/report_period)
                    r = 0
                    p, r, f = self.evaluate()
                    #self.eval_rewards.append(eval_reward)
                    self.precision.append(p)
                    self.recall.append(r)
                    self.fhalf.append(f)
        except:
            print("Exception has occured, saving models now...")
            #print("Exception due to", e)
            traceback.print_exc()
        finally:
            reward.extend(self.reward) #agglomerate historic rewards
            #eval_rewards.extend(self.eval_rewards)
            precision.extend(self.precision)
            recall.extend(self.recall)
            fhalf.extend(self.fhalf)
            self.reward = reward
            #self.eval_rewards = eval_rewards
            self.precision = precision
            self.recall = recall
            self.fhalf = fhalf

            torch.save(self.agent.critic.q_target_decoder.state_dict(), 'q_test/q_target_decoder.pt')
            torch.save(self.agent.critic.q_decoder.state_dict(), 'q_test/q_decoder.pt')
            torch.save(self.agent.critic.optimizer.state_dict(), 'q_test/optimizer.pt')
            torch.save(self.agent.critic.learning_rate_scheduler.state_dict(), 'q_test/learning_rate_scheduler.pt')
            torch.save({'epoch': epoch_trained + i, 'reward': self.reward, 'precision':self.precision, 'recall':self.recall, 'fhalf':self.fhalf, 't':self.agent.t, 'num_param_updates': self.agent.num_param_updates}, 'q_test/misc.pt')
            torch.save({"next_idx":self.agent.replay_buffer.next_idx, 
                        "num_in_buffer":self.agent.replay_buffer.num_in_buffer, 
                        "obs":self.agent.replay_buffer.obs, 
                        "action":self.agent.replay_buffer.action, 
                        "reward":self.agent.replay_buffer.reward, 
                        "done":self.agent.replay_buffer.done}, 'q_test/replay_buffer.pt')
            torch.save(self.agent.env.encoder.state_dict(),'q_test/env_encoder.pt')
            torch.save(self.agent.env.decoder.state_dict(),'q_test/env_decoder.pt')
            print("Trained {} iterations in total".format(epoch_trained + i))
        

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
        gold, edit, true_edit = 0, 0, 0
        precision, recall, fhalf = 0, 0, 0
        for t in self.agent.env.test_data:
            src = t[0]
            trg = t[1]
            test_obs = self.agent.env.reset(True, t)
            translated = []
            for i in range(len(trg)):
                action = self.agent.actor.get_actions(test_obs)
                action_word = self.agent.env.lang.index2word[action[0]]
                # print("src: " + str(src[i]))
                # print("trg: " + str(trg[i]))
                # print("action: " + str(action[0]))
                if src[i] != trg[i]:
                    gold += 1
                if src[i] != action_word:
                    edit += 1
                if src[i] != trg[i] and src[i] != action_word and action_word == trg[i]:
                    true_edit += 1
                obs, reward, done = self.agent.env.step(test_obs, action)
                translated.append(self.agent.env.lang.index2word[action.item()])
                test_obs = obs
                steps += 1
                r += reward
        precision = true_edit / edit
        recall = true_edit / gold
        fhalf = (1 + 0.5**2) * precision * recall / (recall + 0.5**2 * precision)
        print("should and did edit {}, should edit {}, did edit {}".format(true_edit, gold, edit))
        print("precision: " + str(precision))
        print("recall: " + str(recall))
        print("fhalf: " + str(fhalf))
        return precision, recall, fhalf
            #print('=', src)
            #print('<', trg)
            #print('>', ''.join(translated))
            #print('')

class WeakEnvironment(object):
    def __init__(self, train_data, test_data):
        self.encoder = EncoderRNN()
        try:
          self.encoder.load_state_dict(torch.load('q_test/env_encoder.pt'))
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
          self.decoder.load_state_dict(torch.load('q_test/env_decoder.pt'))
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        action_cur = torch.tensor([[action[0]]]).to(self.device)
        prev_hidden = prev_hidden.to(self.device)
        decoded_result = self.decoder(action_cur, prev_hidden, encoder_padded)
        next_hidden = ptu.to_numpy(decoded_result[1].detach())

        # the reward can't be too small, otherwise no signal
        # the reward can't be too large, otherwise will only learn little to be satisified
        # a reward of x means that 1 correct prediction will be killed by x incorrect predictions
        if (action == target_id[curr_index]):
            reward = 10 #5/(((abs(l)**3)+1e-5) + 0.05)
        else:
            reward = -1 #
        assert len(target_id) == len(observation[0]), observation[0]
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

        src_id = torch.tensor(src_id).to(self.device)
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

class DQNAgent(object):
    def __init__(self, params):
        self.t = params['t']
        self.exploration = params['exploration_schedule']
        self.batch_size = params['batch_size']
        self.replay_buffer = ReplayBuffer(params['replay_buffer_size'], params['frame_history_len'], params['replay_buffer'])
        self.learning_starts = params['learning_starts']
        self.learning_freq = params['learning_freq']
        self.num_param_updates = params['num_param_updates']
        self.target_update_freq = params['target_update_freq']
        self.optimizer_spec = params['optimizer_spec']
        self.env = WeakEnvironment(params['train'], params['test'])
        self.critic = DQNCritic(params['critic_params'], self.optimizer_spec, self.env)
        self.last_obs = self.env.reset()
        self.actor = ArgMaxPolicy(self.critic, self.t, self.env.action_space, self.env.lang)
    
    def step(self):
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        eps = self.exploration.value(self.t)
        is_random = ((np.random.random() <= eps) or (self.t < self.learning_starts))
        if not is_random:
            recent_obs = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_actions(recent_obs)
        else:
            action = self.env.random_actions()
        
        #print('>', self.last_obs[0])
        target_id = self.env.encoder.input_ids[self.last_obs[0]][1]
        target_prev = SOS_token if self.last_obs[4] == 0 else target_id[self.last_obs[4] - 1]
        target_curr = target_id[self.last_obs[4]]
        #print('=', ''.join([self.env.encoder.lang.index2word[target_prev], self.env.encoder.lang.index2word[target_curr]]))
        #print('<', ''.join([self.env.encoder.lang.index2word[self.last_obs[3][0]], self.env.encoder.lang.index2word[action.item()]]))
        #print(self.t, self.replay_buffer.can_sample(self.batch_size))
        #print('')

        obs, reward, done = self.env.step(self.last_obs, action)
        self.last_obs = obs
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)
        self.actor.seen_action[action[0]] += 1 # update the number of times seen

        if done:
            self.last_obs = self.env.reset()
        return reward

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]
        
    def train(self, ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch):
        if (self.t > self.learning_starts and self.t % self.learning_freq == 0 and self.replay_buffer.can_sample(self.batch_size)):
            self.critic.update(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)

            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        self.actor.t = self.t

class DQNCritic(object):
    def __init__(self, params, optimizer_spec, env):
        self.env = env
        self.encoder = self.env.encoder
        self.decoder = self.env.decoder
        # the env decoder is used to generate the hidden states required for the state vector. 
        # should be independent from q networks, and should not be updated since it is the env
        self.q_target_decoder = AttnDecoder(self.encoder.hidden_size, self.encoder.input_size)
        self.q_decoder = AttnDecoder(self.encoder.hidden_size, self.encoder.input_size)
        try:
          self.q_target_decoder.load_state_dict(torch.load('q_test/q_target_decoder.pt'))
          self.q_target_decoder.train()
          self.q_decoder.load_state_dict(torch.load('q_test/q_decoder.pt'))
          self.q_decoder.train()
          print("奥利给!Model Loaded!")
        except Exception as e:
          print("Attempting to load q model but failed due to", e)
         
        self.loss = nn.SmoothL1Loss()
        self.grad_norm_clipping = params['grad_norm_clipping']
        self.optimizer_spec = optimizer_spec
        self.optimizer = self.optimizer_spec.constructor(
            self.q_decoder.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        try: 
          self.optimizer.load_state_dict(torch.load('q_test/optimizer.pt'))
          print("奥利给!Optimizer Loaded!")
        except Exception as e:
          print("Attempting to load q optimizer but failed due to", e)
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        try: 
          self.learning_rate_scheduler.load_state_dict(torch.load('q_test/learning_rate_scheduler.pt'))
          print("奥利给!Learning rate scheduler Loaded!")
        except Exception as e:
          print("Attempting to load q learning rate scheduler but failed due to", e)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
          self.encoder = self.encoder.cuda()
          self.decoder = self.decoder.cuda()
          self.q_decoder = self.q_decoder.cuda()
          self.q_target_decoder = self.q_target_decoder.cuda()
        
          # has to do this when use load_state_dict with optimizer https://github.com/pytorch/pytorch/issues/2830
          for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()



    def q_net_target(self, ob):
        encoder_padded = ptu.from_numpy(np.array(ob[:,1].tolist()).astype(np.float32))[:,0,:,:]
        decoder_hidden = ptu.from_numpy(np.array(ob[:,2].tolist()).astype(np.float32))[:,0,:,:]
        decoder_input = ptu.from_numpy(np.array(ob[:,3].tolist()).astype(np.float32)).long()

        encoder_padded = encoder_padded.to(self.device)
        decoder_hidden = decoder_hidden.to(self.device)
        decoder_input = decoder_input.to(self.device)
        output, _, _ = self.q_target_decoder(decoder_input, decoder_hidden, encoder_padded)
        return output.squeeze()
    
    def q_net(self, ob):
        encoder_padded = ptu.from_numpy(np.array(ob[:,1].tolist()).astype(np.float32))[:,0,:,:]
        decoder_hidden = ptu.from_numpy(np.array(ob[:,2].tolist()).astype(np.float32))[:,0,:,:]
        decoder_input = ptu.from_numpy(np.array(ob[:,3].tolist()).astype(np.float32)).long()

        decoder_input = decoder_input.to(self.device)
        decoder_hidden = decoder_hidden.to(self.device)
        encoder_padded = encoder_padded.to(self.device)
        output, _, _ = self.q_decoder(decoder_input, decoder_hidden, encoder_padded)
        return output.squeeze()

    def update(self, ob_no, ac_na, reward_n, next_ob_no, terminal_n):
        # everything else should be numpy arrays up til this point
        ob_no = np.array(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = np.array(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        ac_na = ac_na.to(self.device)
        q = torch.gather(self.q_net(ob_no), 1, ac_na.unsqueeze(1)).squeeze()
        #print('q', q.shape)
        ac_qmax = torch.argmax(self.q_net(next_ob_no), dim=1).unsqueeze(1)

        # next_ob_no = next_ob_no.to(self.device)
        q_target = self.q_net_target(next_ob_no)
        q_target_plug_in = q_target.gather(1, ac_qmax).squeeze()
        terminal_n = terminal_n.to(self.device)
        reward_n = reward_n.to(self.device)
        target = reward_n + q_target_plug_in*(torch.logical_not(terminal_n)).detach()

        loss = self.loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_decoder.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        
    def update_target_network(self):
        for target_param, param in zip(self.q_target_decoder.parameters(), self.q_decoder.parameters()):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values) 

class ArgMaxPolicy(object):

    def __init__(self, critic, t, action_space, lang):
        self.critic = critic
        self.t = t
        self.action_space = action_space
        self.seen_action = np.array([1e-6 for _ in range(len(action_space))])
        self.lang = lang

    def get_actions(self, obs):
        observation = np.array(obs, dtype=object).reshape(-1,5)
        
        qval = self.critic.qa_values(observation)
        action_variance = np.sqrt(2*np.log(self.t)/self.seen_action)
        qval = qval+action_variance
        batch_size = len(observation)

        next_pos_to_predict = observation[:,4].tolist()
        src = observation[:,0].tolist()
        #print(self.t, src, next_pos_to_predict)
        next_id_in_src = [self.lang.word2index[src[i][next_pos_to_predict[i]]] for i in range(batch_size)]
        # you have to allow itself to be predicted as well
        easily_confused = [self.lang.correct_confused[id] for id in next_id_in_src]
        qval_of_interest = [(easily_confused[i],qval[easily_confused[i]]) for i in range(batch_size)]
        action = np.array([q[0][np.argmax(q[1])] for q in qval_of_interest])

        return action

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len, params):
        self.size = size
        self.frame_history_len = frame_history_len
        self.next_idx      = params['next_idx']
        self.num_in_buffer = params['num_in_buffer']

        self.obs = params['obs']
        self.action = params['action']
        self.reward = params['reward']
        self.done = params['done']

    def can_sample(self, batch_size):
        return batch_size + 1 <= self.num_in_buffer

    def sample_n_unique(self, sampling_f, n):
        res = []
        while len(res) < n:
            candidate = sampling_f()
            if candidate not in res:
                res.append(candidate)
        return res
    
    def sample(self, batch_size):
        idxes = self.sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self.encode_sample(idxes)

    def encode_sample(self, idxes):
        obs_batch      = [self.encode_observation(idx) for idx in idxes]
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = [self.encode_observation(idx + 1) for idx in idxes]
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask

    def store_frame(self, frame):
        if self.obs is None:
            self.obs      = np.empty([self.size] + list((1, len(frame)))).tolist()
            self.action   = np.empty([self.size], dtype=np.int32)
            self.reward   = np.empty([self.size], dtype=np.float32)
            self.done     = np.empty([self.size], dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames."""
        assert self.num_in_buffer > 0
        return self.encode_observation((self.next_idx - 1) % self.size)

    def encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        
        return self.obs[end_idx-1]

    def store_effect(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done 

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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        encoder_outputs = encoder_outputs.to(self.device)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        #print(embedded.size(), attn_applied.size())

        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)

        output = F.relu(output)
        #print(output.size(), hidden.size())
        output, hidden = self.gru(output.permute(1,0,2), hidden.permute(1,0,2))
        #print(output.size(), hidden.size())

        output = self.out(output)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

OptimizerSpec = namedtuple(
    "OptimizerSpec",
    ["constructor", "optim_kwargs", "learning_rate_schedule"],
)
class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()

class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v

def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)