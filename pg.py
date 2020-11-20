import numpy as np
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
import pytorch_utils as ptu
import pandas as pd

SOS_token = 0#101
EOS_token = 1#102
MAX_LENGTH = 12 # take into account the CLS&SEP that will be added later
class BertPolicyGradientAlgo(object):
    def __init__(self, n_iter, n_traj, data):
        self.n_iter = n_iter
        self.n_traj = n_traj
        self.env_max_step = MAX_LENGTH
        self.data = data
        self.encoder = ChinBERT()
        self.input_ids = self.encoder.embed(data) # cast plaintext to our environment space (input_id)
        print("conversion to input ids done...")
        self.decoder = AttnDecoder(self.encoder.config.hidden_size, self.encoder.tokenizer.vocab_size)
        self.criterion = nn.NLLLoss()
        self.rewards = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plot_losses = []

    def run(self):
        for i in range(self.n_iter):
            print("cycle iteration {}...".format(i))
            # collect trajectories
            trajectories = self.collectTrajectories(self.n_traj)
            # get metric
            self.getRewardMetric(trajectories)
            # update actor
            self.updateActors(trajectories)
    
    def getRewardMetric(self, trajectories):
        rewards_curr_iter = []
        for t in trajectories:
            rewards_curr_iter.append(np.sum(t.rewards, axis=0).reshape(1,-1))
        self.rewards.append(np.mean(np.array(rewards_curr_iter), axis=0))
        

    def collectTrajectories(self, n_traj):
        trajectories = []
        plot_loss_total = 0
        for _ in range(n_traj):
            traj = self.collectTrajectory()
            trajectories.append(traj[0])
            plot_loss_total += traj[1]
        self.plot_losses.append(plot_loss_total/10)
        return np.array(trajectories)
    
    def collectTrajectory(self):
        #self.encoder.optimizer.zero_grad()
        #self.decoder.optimizer.zero_grad()
        states, actions, rewards = [[], [], []], [], []

        training_pair = random.choice(self.data)
        src_plain = training_pair[0]
        target_plain = training_pair[1]
        src_id = self.input_ids[src_plain][0]
        target_id = self.input_ids[src_plain][1]
        encoder_output = self.encoder(src_plain, 0)
        encoder_sequence_outputs = encoder_output[0].squeeze()
        encoder_pooled_output = encoder_output[1].reshape((1,1,-1))
        target_length = len(target_id)

        #TODO:len target is max, could add terminal vs. finish
        encoder_padded = torch.zeros(1, self.env_max_step, self.decoder.hidden_size)
        encoder_padded[:,:len(encoder_sequence_outputs),:] = encoder_sequence_outputs

        loss = 0
        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_pooled_output
        translated_sentence = []
        for i in range(1, target_length):
            states[0].append(decoder_input.tolist())
            states[1].append(decoder_hidden.tolist())
            states[2].append(encoder_padded.tolist())

             
            action_distribution, output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_padded)
            action = action_distribution.sample()
            # 1/x to encourage close prediction a lot more
            actions.append(action)
            #print(output.shape, output[0].shape)
            #l = self.criterion(output[0], torch.tensor([target_id[i]])).item()
            l = output[0][0][target_id[i]].item()
            loss += self.criterion(output[0], torch.tensor([target_id[i]]))
            #print(l, 20/(abs(l)+1e-5))
            if (action.item() == target_id[curr_index]):
                reward = 10 #5/(((abs(l)**3)+1e-5) + 0.05)
            else:
                reward = -1 #
            rewards.append(20/(abs(l)+1e-5)) # your problem is that you don't see correct actions less
            decoder_input = torch.tensor(action)
            translated_sentence.append(action)

        #loss.backward()

        #self.decoder.optimizer.step()
        #self.encoder.optimizer.step()

        #print('>', src_plain)
        #print('=', target_plain)
        #print('<', ''.join(self.encoder.tokenizer.convert_ids_to_tokens(translated_sentence)))
        #print('')


        
        states[0] = np.array(states[0]).astype(np.int32) 
        states[1] = np.array(states[1]).astype(np.float32).reshape((-1, 1, self.decoder.hidden_size)) 
        states[2] = np.array(states[2]).astype(np.float32) 
        return Trajectory(states, np.squeeze(np.array(actions)), np.squeeze(np.array(rewards))), loss.item() / target_length
    

    def updateActors(self, trajectories):
        loss = torch.zeros(1)
        for t in trajectories:
            #print(t.observations.shape, t.actions.shape, t.rewards.shape)
            obs = t.observations
            decoder_input, decoder_hidden, encoder_padded = torch.from_numpy(obs[0]).long()[:,:,0], ptu.from_numpy(obs[1]), ptu.from_numpy(obs[2]).squeeze()
            acs = ptu.from_numpy(t.actions)
            # actions could be squeezed, need to reshape to 1*N so log prob calculates respectively instead of a matrix for categorical distirbution
            # N*1 for normal batch
            acs = torch.reshape(acs, (1,-1)) 
            
            action_distribution, _, _, _ = self.decoder(decoder_input, decoder_hidden, encoder_padded)

           
            neg_log_prob = -1*action_distribution.log_prob(acs)
            neg_log_prob = torch.squeeze(neg_log_prob)

            causality_cumsum = np.flip(np.cumsum(np.flip(t.rewards))).copy()
            traj_reward = ptu.from_numpy(causality_cumsum) # causality trick
            loss += torch.dot(neg_log_prob, traj_reward)

        self.decoder.optimizer.zero_grad()
        self.encoder.optimizer.zero_grad()
        loss.backward()
        self.decoder.optimizer.step()
        self.encoder.optimizer.step()

class NoBertPolicyGradientAlgo(object):
    def __init__(self, n_iter, n_traj, data, test_data):
        self.t = 0
        self.n_iter = n_iter
        self.n_traj = n_traj
        self.env_max_step = MAX_LENGTH
        self.test_data = test_data
        self.data = data
        self.encoder = EncoderRNN()
        self.action_space = [i for i in range(self.encoder.input_size)]
        print("action space size: ", len(self.action_space))
        self.seen_action = np.array([1e-6 for _ in range(len(self.action_space))])
        self.input_ids = self.encoder.embed([data, test_data]) # cast plaintext to our environment space (input_id)
        self.lang = self.encoder.lang
        print("conversion to input ids done...")
        self.decoder = AttnDecoder(self.encoder.hidden_size, self.encoder.input_size)
        if torch.cuda.is_available():
          self.encoder = self.encoder.cuda()
          self.decoder = self.decoder.cuda()
        self.criterion = nn.NLLLoss()
        self.rewards = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plot_losses = []
        self.eval_losses = []

    def run(self):
        for i in range(self.n_iter):
            print("cycle iteration {}...".format(i))
            self.t = i
            # collect trajectories
            trajectories = self.collectTrajectories(self.n_traj)
            # get metric
            self.getRewardMetric(trajectories)
            # update actor
            try: 
                self.updateActors(trajectories)
            except:
                continue

    
    def getRewardMetric(self, trajectories):
        rewards_curr_iter = []
        for t in trajectories:
            rewards_curr_iter.append(np.sum(t.rewards, axis=0).reshape(1,-1))
        self.rewards.append(np.mean(np.array(rewards_curr_iter), axis=0))
        
    def collectTrajectories(self, n_traj):
        trajectories = []
        plot_loss_total = 0
        eval_loss = 0
        for _ in range(n_traj):
            traj = self.collectTrajectory()
            trajectories.append(traj[0])
            plot_loss_total += traj[1]
        self.plot_losses.append(plot_loss_total/n_traj)

        eval_batch = 100
        for _ in range(eval_batch):
            loss = self.evaluate()
            eval_loss += loss
        self.eval_losses.append(eval_loss/eval_batch)
        return np.array(trajectories)

    def evaluate(self):
        training_pair = random.choice(self.test_data)
        src_plain = training_pair[0]
        target_plain = training_pair[1]
        src_id = self.input_ids[src_plain][0]
        target_id = self.input_ids[src_plain][1]
        input_length = len(src_id)
        target_length = len(target_id)

        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = torch.zeros(self.env_max_step, self.encoder.hidden_size, device=self.device)
        src_id = torch.tensor(src_id)
        src_id = src_id.to(self.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(src_id[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        encoder_padded = torch.zeros(1, self.env_max_step, self.decoder.hidden_size)
        encoder_padded[:,:len(encoder_outputs),:] = encoder_outputs
        
        decoder_input = torch.tensor([[SOS_token]], device=self.device)
        decoder_hidden = encoder_hidden

        translated_sentence = []
        loss = 0

        for i in range(target_length):
            #print(decoder_input.shape, decoder_hidden.shape, encoder_padded.shape)
            decoder_input = decoder_input.to(self.device)
            action_distribution, output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_padded)
            action_variance = np.sqrt(2*np.log(self.t)/self.seen_action)
            #weighted_output = output.cpu().detach().numpy()+action_variance
            next_id_in_src = src_id[i].item()
            easily_confused = self.lang.confused[next_id_in_src]+[next_id_in_src]
            output_of_interest = (easily_confused,output[:,:,easily_confused])
            action = torch.tensor([[output_of_interest[0][torch.argmax(output_of_interest[1], dim=2)]]])
            #action = torch.tensor([[torch.argmax(output)]])
            #while action.item() == 0 or action.item() == 1:
            #    action = action_distribution.sample() # won't allow SOS & EOS mid sentence
            target_id_cur = torch.tensor([target_id[i]]).to(self.device)
            loss += self.criterion(output[0], target_id_cur)
            
            decoder_input = torch.tensor(action)
            translated_sentence.append(action)

        #print('>', src_plain)
        #print('=', target_plain)
        #print('<', ''.join([self.lang.index2word[w.item()] for w in translated_sentence]))
        #print('')
        
        return loss.item() / target_length

    
    def collectTrajectory(self):
        self.encoder.optimizer.zero_grad()
        self.decoder.optimizer.zero_grad()
        states, actions, rewards = [[], [], []], [], []

        training_pair = random.choice(self.data)
        src_plain = training_pair[0]
        target_plain = training_pair[1]
        src_id = self.input_ids[src_plain][0]
        target_id = self.input_ids[src_plain][1]
        input_length = len(src_id)
        target_length = len(target_id)

        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = torch.zeros(self.env_max_step, self.encoder.hidden_size, device=self.device)
        
        src_id = torch.tensor(src_id).to(self.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(src_id[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        encoder_padded = torch.zeros(1, self.env_max_step, self.decoder.hidden_size)
        encoder_padded[:,:len(encoder_outputs),:] = encoder_outputs
        
        decoder_input = torch.tensor([[SOS_token]], device=self.device)
        decoder_hidden = encoder_hidden

        translated_sentence = []
        loss = 0
        
        for i in range(target_length):
            states[0].append(decoder_input.tolist())
            states[1].append(decoder_hidden.tolist())
            states[2].append(encoder_padded.tolist())

            decoder_input = decoder_input.to(self.device)
            action_distribution, output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_padded)

            action_variance = np.sqrt(2*np.log(self.t)/self.seen_action)

            weighted_output = output.cpu().detach().numpy()+action_variance
            
            # it is very hard here because there isn't really a right direction! it is either correct or completely wrong.
            # your problem is that you don't see correct actions often because all of the others are completely wrong
            # 1/x to encourage close prediction a lot more
            
            next_id_in_src = src_id[i].item()
            easily_confused = self.lang.confused[next_id_in_src]+[next_id_in_src]
            output_of_interest = (easily_confused,output[:,:,easily_confused])
            action = torch.tensor([[output_of_interest[0][torch.argmax(output_of_interest[1], dim=2)]]])
            #action = torch.tensor([[torch.argmax(output)]])#action_distribution.sample()#torch.tensor([[torch.argmax(weighted_output).item()]])#
            #while action.item() == 0 or action.item() == 1:
            #    action = action_distribution.sample() # won't allow SOS & EOS mid sentence
            actions.append(action)
            self.seen_action[action.item()] += 1
            
            target_id_cur = torch.tensor([target_id[i]]).to(self.device)
            l = self.criterion(output[0], target_id_cur).item()
            #reward = 1/(l + 1e-5)
            if (action.item() == target_id[i]):
                reward = 10 #5/(((abs(l)**3)+1e-5) + 0.05)
            else:
                reward = -1 #
            #l = output[0][0][target_id[i]].item()
            #reward = 5/((abs(l)**3)+1e-5) + 0.05
            rewards.append(reward)

            loss += self.criterion(output[0], target_id_cur)
            #if np.exp(l) > 0.3: print(self.lang.index2word[action.item()], target_plain[i])
            
            decoder_input = torch.tensor([[target_id[i]]])#torch.tensor(action)
            translated_sentence.append(action)

        loss.backward()

        self.decoder.optimizer.step()
        self.encoder.optimizer.step()
        #print('>', src_plain)
        #print('=', target_plain)
        #print('<', ''.join([self.lang.index2word[w.item()] for w in translated_sentence]))
        #print('')
        
        states[0] = np.array(states[0]).astype(np.int32) 
        states[1] = np.array(states[1]).astype(np.float32).reshape((-1, 1, self.decoder.hidden_size)) 
        states[2] = np.array(states[2]).astype(np.float32) 
        return Trajectory(states, np.squeeze(np.array(actions)), np.squeeze(np.array(rewards))), loss.item() / target_length
    

    def updateActors(self, trajectories):
        self.decoder.optimizer.zero_grad()
        self.encoder.optimizer.zero_grad()
        loss = torch.zeros(1)
        for t in trajectories:
            obs = t.observations
            decoder_input, decoder_hidden, encoder_padded = ptu.from_numpy(obs[0]).long()[:,:,0], ptu.from_numpy(obs[1]), ptu.from_numpy(obs[2]).squeeze()
            acs = ptu.from_numpy(t.actions)

            # actions could be squeezed, need to reshape to 1*N so log prob calculates respectively instead of a matrix for categorical distirbution
            # N*1 for normal batch
            acs = torch.reshape(acs, (1,-1)) 
            
            action_distribution, _, _, _ = self.decoder(decoder_input, decoder_hidden, encoder_padded)

           
            neg_log_prob = -1*action_distribution.log_prob(acs)
            neg_log_prob = torch.squeeze(neg_log_prob)

            causality_cumsum = np.flip(np.cumsum(np.flip(t.rewards))).copy()
            traj_reward = ptu.from_numpy(causality_cumsum) # causality trick
            loss += torch.dot(neg_log_prob, traj_reward)

        
        loss.backward()
        self.decoder.optimizer.step()
        self.encoder.optimizer.step()


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.next_index = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.next_index
            self.index2word[self.next_index] = word
            self.next_index += 1 
    
    def addConfusion(self):
      self.confused = {key: [] for key in self.index2word.keys()} 
      f = open('confusion.txt',"r")
      for line in f:
          if line[0] in self.word2index.keys():
              key = self.word2index[line[0]]
              confusions = []
              for w in line[2:-1]:
                  if w in self.word2index.keys():
                    confusions.append(self.word2index[w])
              self.confused[key] = confusions

class ChinBERT(nn.Module):
    def __init__(self, dropout=0.3, lr=0.005):
        super(ChinBERT, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")#("hfl/chinese-bert-wwm")
        self.config = BertConfig.from_pretrained("bert-base-chinese")#("hfl/chinese-bert-wwm")
        self.bert = BertModel.from_pretrained("bert-base-chinese")#("hfl/chinese-bert-wwm")
        self.drop = nn.Dropout(p=dropout)
        self.input_ids = {}
        self.attn_masks = {}
        self.optimizer = optim.Adam(self.parameters())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def embed(self, dataset):
        for d in dataset:
            encoding0 = self.tokenizer.encode_plus(
                d[0],
                max_length=MAX_LENGTH,
                add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                return_token_type_ids=False,
                padding=True,
                return_attention_mask=True,
                return_tensors='np',  # Return PyTorch tensors
            )
            encoding1 = self.tokenizer.encode_plus(
                d[1],
                max_length=MAX_LENGTH,
                add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                return_token_type_ids=False,
                padding=True,
                return_attention_mask=True,
                return_tensors='np',  # Return PyTorch tensors
            )
            encodings = [encoding0, encoding1]
            self.input_ids[d[0]] = [e['input_ids'][0] for e in encodings]
            self.attn_masks[d[0]] = [e['attention_mask'][0] for e in encodings] 
        return self.input_ids
    def forward(self, input_sentence, is_target, is_train=True):
        input_id, attn_mask = self.input_ids[input_sentence][is_target], self.attn_masks[input_sentence][is_target]
        input_id, attn_mask = torch.from_numpy(input_id).reshape(1,-1), torch.from_numpy(attn_mask).reshape(1,-1)
        if is_train:
            self.bert.train()
        else:
            self.bert.eval()
        sequence_output, pooled_output = self.bert(
            input_ids=input_id,
            attention_mask=attn_mask
        )
        
        return sequence_output, pooled_output

class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.lang = None
        self.prepareData()
        self.lang.addConfusion()
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
        
        df = pd.read_csv("data/sighan10.csv")
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
                #self.attn_masks[d[0]] = [e['attention_mask'][0] for e in encodings] 
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

        encoder_outputs = encoder_outputs.to(self.device)        
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), 2)), dim=2)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        #print(embedded.size(), attn_applied.size())

        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)

        output = F.relu(output)
        #print(output.size(), hidden.size())
        output, hidden = self.gru(output.permute(1,0,2), hidden.permute(1,0,2))
        #print(output.size(), hidden.size())

        output = F.log_softmax(self.out(output), dim=2)
        action_distribution = torch.distributions.Categorical(probs=torch.exp(output))
        return action_distribution, output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class Trajectory(object):
    def __init__(self, observations, actions, rewards):
        """
            Each argument is a list of what it contains
        """
        self.observations = observations
        self.actions = actions
        self.rewards = rewards