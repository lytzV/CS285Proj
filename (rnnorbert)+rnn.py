# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
from transformers import BertTokenizer, BertModel, BertConfig

import unicodedata
import string
import re
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from gensim.models import Word2Vec


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 101
EOS_token = 102
MAX_LENGTH = 52 # take into account the CLS&SEP that will be added later


def loadData():
    df = pd.read_csv("data/sighan10.csv")
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
        
        df = pd.read_csv("data/data50.csv")
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

    def embed(self, dataset):
        for d in dataset:
            encodings = self.indexesFromPair(self.lang, d)
            self.input_ids[d[0]] = encodings
            #self.attn_masks[d[0]] = [e['attention_mask'][0] for e in encodings] 
        return self.input_ids

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, learning_rate=0.01, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, input, hidden, encoder_outputs):
        
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        
        output = F.relu(output)
        #print(output.shape, hidden.shape)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        action_distribution = torch.distributions.Categorical(probs=torch.exp(output))
        return action_distribution, output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class BatchAttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, learning_rate=0.01, max_length=MAX_LENGTH):
        super(BatchAttnDecoder, self).__init__()
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

def train(input_sentence, target_sentence, encoder, decoder, criterion, max_length=MAX_LENGTH):
    encoder.optimizer.zero_grad()
    decoder.optimizer.zero_grad()

    # BERT 
    """#encoder_outputs = encoder(input_sentence, 0)
    #encoder_sequence_outputs = encoder_outputs[0].squeeze()
    #encoder_pooled_output = encoder_outputs[1].reshape((1,1,-1))
    #target_input_ids = encoder.input_ids[input_sentence][1]
    #target_length = len(target_input_ids)
    
    #encoder_padded = torch.zeros(max_length, decoder.hidden_size)
    #try:
    #    encoder_padded[:len(encoder_sequence_outputs),:] = encoder_sequence_outputs
    #except:
    #    print(input_sentence, target_sentence, encoder_sequence_outputs.size(), target_length)
    #    return
    loss = 0
    decoder_input = torch.tensor([[SOS_token]])
    decoder_hidden = encoder_pooled_output
    translated_sentence = []
    for i in range(1, target_length):
        action_distribution, decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_padded)
        loss += criterion(decoder_output, torch.tensor([target_input_ids[i]]))
        action = action_distribution.sample()
        decoder_input = torch.tensor(action)#torch.tensor(torch.argmax(decoder_output)) #torch.tensor([target_input_ids[i]])
        translated_sentence.append(action)
    
    loss.backward()

    decoder.optimizer.step()
    encoder.optimizer.step()

    print('>', input_sentence)
    print('=', target_sentence)
    print('<', ''.join(encoder.tokenizer.convert_ids_to_tokens(translated_sentence)))
    print('')
    return loss.item() / target_length"""

    src_plain = input_sentence
    target_plain = target_sentence
    src_id = encoder.input_ids[src_plain][0]
    target_id = encoder.input_ids[src_plain][1]
    input_length = len(src_id)
    target_length = len(target_id)

    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(torch.tensor(src_id[ei]),encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]

    encoder_padded = torch.zeros(1, MAX_LENGTH, decoder.hidden_size)
    encoder_padded[:,:len(encoder_outputs),:] = encoder_outputs

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    translated_sentence = []
    loss = 0
    for i in range(target_length):
        #print(decoder_input.shape, decoder_hidden.shape, encoder_padded.shape)
        action_distribution, output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_padded)
        loss += criterion(output[0], torch.tensor([target_id[i]]))
        next_id_in_src = src_id[i].item()
        easily_confused = encoder.lang.confused[next_id_in_src]+[next_id_in_src]
        output_of_interest = (easily_confused,output[:,:,easily_confused])
        action = torch.tensor([[output_of_interest[0][torch.argmax(output_of_interest[1], dim=2)]]])
        #action = action_distribution.sample()
        decoder_input = torch.tensor(action) #torch.tensor([[target_id[i]]]) #torch.tensor(action)#torch.tensor(torch.argmax(decoder_output)) torch.tensor([[target_id[i]]])
        translated_sentence.append(action)

    loss.backward()

    decoder.optimizer.step()
    encoder.optimizer.step()

    #print('>', input_sentence)
    #print('=', target_sentence)
    #print('<', ''.join([encoder.lang.index2word[w.item()] for w in translated_sentence]))
    #print('')
    return loss.item() / target_length
    
    

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    #loc = ticker.MultipleLocator(base=0.2)
    #ax.yaxis.set_major_locator(loc)
    for (l,p) in points.items():
        plt.plot(p, label=l)
    plt.legend()
    plt.show()

def trainIters(encoder, decoder, criterion, n_iters, pairs, test_pairs, print_every=1000, plot_every=10, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    eval_losses = []

    for iter in range(1, n_iters + 1):
        eval_loss = 0
        training_pair = random.choice(pairs)
        input_sentence = training_pair[0]
        target_sentence = training_pair[1]

        loss = train(input_sentence, target_sentence, encoder, decoder, criterion)
       
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            for _ in range(100):
                eval_pair = random.choice(test_pairs)
                input_sentence = eval_pair[0]
                target_sentence = eval_pair[1]
                loss = evaluate(input_sentence, target_sentence, encoder, decoder, criterion)

                eval_loss += loss
            eval_losses.append(eval_loss/100)

    return plot_losses, eval_losses

def evaluate(input_sentence, target_sentence, encoder, decoder, criterion, max_length=MAX_LENGTH):
    # BERT 
    """#encoder_outputs = encoder(input_sentence, 0)
    #encoder_sequence_outputs = encoder_outputs[0].squeeze()
    #encoder_pooled_output = encoder_outputs[1].reshape((1,1,-1))
    #target_input_ids = encoder.input_ids[input_sentence][1]
    #target_length = len(target_input_ids)
    
    #encoder_padded = torch.zeros(max_length, decoder.hidden_size)
    #try:
    #    encoder_padded[:len(encoder_sequence_outputs),:] = encoder_sequence_outputs
    #except:
    #    print(input_sentence, target_sentence, encoder_sequence_outputs.size(), target_length)
    #    return
    loss = 0
    decoder_input = torch.tensor([[SOS_token]])
    decoder_hidden = encoder_pooled_output
    translated_sentence = []
    for i in range(1, target_length):
        action_distribution, decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, decoder_hidden, encoder_padded)
        loss += criterion(decoder_output, torch.tensor([target_input_ids[i]]))
        action = action_distribution.sample()
        decoder_input = torch.tensor(action)#torch.tensor(torch.argmax(decoder_output)) #torch.tensor([target_input_ids[i]])
        translated_sentence.append(action)
    
    loss.backward()

    decoder.optimizer.step()
    encoder.optimizer.step()

    print('>', input_sentence)
    print('=', target_sentence)
    print('<', ''.join(encoder.tokenizer.convert_ids_to_tokens(translated_sentence)))
    print('')
    return loss.item() / target_length"""

    src_plain = input_sentence
    target_plain = target_sentence
    src_id = encoder.input_ids[src_plain][0]
    target_id = encoder.input_ids[src_plain][1]
    input_length = len(src_id)
    target_length = len(target_id)

    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(torch.tensor(src_id[ei]),encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]

    encoder_padded = torch.zeros(1, MAX_LENGTH, decoder.hidden_size)
    encoder_padded[:,:len(encoder_outputs),:] = encoder_outputs

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    translated_sentence = []
    loss = 0
    for i in range(target_length):
        #print(decoder_input.shape, decoder_hidden.shape, encoder_padded.shape)
        action_distribution, output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_padded)
        loss += criterion(output[0], torch.tensor([target_id[i]]))
        next_id_in_src = src_id[i].item()
        easily_confused = encoder.lang.confused[next_id_in_src]+[next_id_in_src]
        output_of_interest = (easily_confused,output[:,:,easily_confused])
        action = torch.tensor([[output_of_interest[0][torch.argmax(output_of_interest[1], dim=2)]]])
        #action = action_distribution.sample()
        decoder_input = torch.tensor(action) #torch.tensor(action)#torch.tensor(torch.argmax(decoder_output)) #torch.tensor([target_input_ids[i]])
        translated_sentence.append(action)

    #print('>', input_sentence)
    #print('=', target_sentence)
    #print('<', ''.join([encoder.lang.index2word[w.item()] for w in translated_sentence]))
    #print('')
    return loss.item() / target_length
    

import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np




if __name__ == "__main__":
    training_data, test_data = loadData()

    #encoder = ChinBERT()
    #assert encoder.tokenizer.vocab_size == 21128, encoder.tokenizer.vocab_size
    #encoder.embed(training_data)
    #decoder = AttnDecoderRNN(encoder.config.hidden_size, encoder.tokenizer.vocab_size)
    encoder = EncoderRNN()
    
    encoder.embed(training_data)
    encoder.embed(test_data)
    decoder = BatchAttnDecoder(encoder.hidden_size, encoder.input_size)


    train_loss, eval_loss = trainIters(encoder, decoder, nn.NLLLoss(), 5000, training_data, test_data, print_every=10)

    showPlot({"Train": train_loss, "Eval": eval_loss})
    #evaluateRandomly(encoder, decoder, training_data)
    #evaluateRandomly(encoder1, attn_decoder1, test_data)


    #evaluateAndShowAttention("elle a cinq ans de moins que moi .")

    #evaluateAndShowAttention("elle est trop petit .")

    #evaluateAndShowAttention("je ne crains pas de mourir .")

    #evaluateAndShowAttention("c est un jeune directeur plein de talent .")





