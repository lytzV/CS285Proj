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
MAX_LENGTH = 51


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

def prepareData():
    print("Reading Chinese Frequency Corpus")
    df = pd.read_csv("data/common_chinese.csv")
    chinese = Lang("chinese")
    for w in df['CHAR']:
        chinese.addWord(w)
    df = pd.read_csv("data/data.csv")
    for s in df['source']:
        chinese.addSentence(s)
    for s in df['reference']:
        chinese.addSentence(s)
    return chinese

def loadData():
    df = pd.read_csv("data/data50.csv")
    dataset = df.to_numpy()
    np.random.shuffle(dataset)
    split_index = int(len(dataset)*0.9)
    training_data, test_data = dataset[:split_index, :], dataset[split_index:, :]
    return training_data, test_data

class ChinBERT(nn.Module):
    def __init__(self, dropout=0.3, lr=0.005):
        super(ChinBERT, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")
        self.config = BertConfig.from_pretrained("hfl/chinese-bert-wwm")
        self.bert = BertModel.from_pretrained("hfl/chinese-bert-wwm")
        self.drop = nn.Dropout(p=dropout)
        self.input_ids = {}
        self.attn_masks = {}
        self.optimizer = optim.SGD(self.bert.parameters(), lr=lr)

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
        
        return sequence_output

class ChinDecoder(nn.Module):
    def __init__(self, n_layer, d_model, n_head, lr=0.005):
        super(ChinDecoder, self).__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.decoderLayer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head)
        self.decoder = nn.TransformerDecoder(self.decoderLayer, num_layers=n_layer)
        self.optimizer = optim.SGD(self.decoder.parameters(), lr=lr)
        self.criterion = nn.L1Loss() #  lambda x,y: abs(x-y)
    def forward(self, encoder_output, curr_token):
        curr_token = curr_token.reshape((curr_token.size()[1], curr_token.size()[0], self.d_model))
        encoder_output = encoder_output.reshape((encoder_output.size()[1], encoder_output.size()[0], self.d_model))
        return self.decoder(curr_token, encoder_output)

class Translator(nn.Module):
    def __init__(self, n_layers, in_dim, out_dim, hidden_size, lr=0.005):
        super(Translator, self).__init__()

        in_size = in_dim
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_size, hidden_size[i]))
            layers.append(nn.ReLU())
            in_size = hidden_size[i]
        layers.append(nn.Linear(in_size, out_dim))
        layers.append(nn.Softmax())
        self.net = nn.Sequential(*layers)
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
        self.criterion = nn.NLLLoss()
    def forward(self, decoder_output):
        return self.net(decoder_output)


def train(input_sentence, target_sentence, encoder, decoder, translator, max_length=MAX_LENGTH):
    encoder.optimizer.zero_grad()
    decoder.optimizer.zero_grad()
    translator.optimizer.zero_grad()

    encoder_output = encoder(input_sentence, 0)
    encoded_target = encoder.input_ids[input_sentence][1]
    target_length = len(encoded_target)

    loss = 0
    curr_tokens = torch.tensor([SOS_token]).reshape(1,-1)
    embedded_curr_tokens = encoder.bert(input_ids = curr_tokens)[0]
    
    translated_sentence = []
    for i in range(1, target_length):
        next_token = decoder(encoder_output, embedded_curr_tokens)
        translated_distribution = translator(next_token.squeeze()).reshape(1, -1)

        #translated_target = torch.zeros(translated_distribution.size())
        #translated_target[encoded_target[i]] = 1
        #print(translated_distribution.size(), torch.tensor(encoded_target[i]).size())
        loss += translator.criterion(translated_distribution, torch.tensor([encoded_target[i]]))
        
        translated_sentence.append(torch.argmax(translated_distribution))
        embedded_curr_tokens = next_token
        
        # since the encoder has changed
        # encoded_target = encoder(input_sentence, 1) 
        # encoder_output = encoder(input_sentence, 0)
        # if next_token == EOS_token:
        #    break
    
    loss.backward()

    translator.optimizer.step()
    decoder.optimizer.step()
    encoder.optimizer.step()

    print('>', input_sentence)
    print('=', target_sentence)
    print('<', ''.join(encoder.tokenizer.convert_ids_to_tokens(translated_sentence)))
    print('')
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

def trainIters(encoder, decoder, translator, n_iters, pairs, print_every=1000, plot_every=1, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for iter in range(1, n_iters + 1):
        training_pair = random.choice(pairs)
        input_sentence = training_pair[0]
        target_sentence = training_pair[1]

        loss = train(input_sentence, target_sentence, encoder, decoder, translator)
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

    showPlot(plot_losses)

import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()
    

def evaluate(encoder, decoder, input_sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        encoder_output = encoder(input_sentence, is_train=False)

        curr_token = SOS_token

        decoder_input = encoder_output
        decoder_outputs = []

        for di in range(max_length):
            decoder_outputs.append(curr_token)
            if curr_token == EOS_token:
                break
            next_token = decoder(decoder_input, curr_token)
            decoder_outputs.append(next_token)
            curr_token = next_token
        
        output_sentence = encoder.tokenizer.decode(decoder_outputs)
        return output_sentence


def evaluateRandomly(encoder, decoder, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ''.join(output_words)
        print('<', output_sentence)
        print('')

"""def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + [w for w in input_sentence] +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateAndShowAttention(input_sentence)
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)"""




if __name__ == "__main__":
    training_data, test_data = loadData()

    encoder = ChinBERT()
    assert encoder.tokenizer.vocab_size == 21128, encoder.tokenizer.vocab_size
    encoder.embed(training_data)
    decoder = ChinDecoder(6, encoder.config.hidden_size, 8)
    translator = Translator(4, encoder.config.hidden_size, encoder.tokenizer.vocab_size, [6, 16, 32, 32])

    trainIters(encoder, decoder, translator, 100, training_data, print_every=10)
    #evaluateRandomly(encoder, decoder, training_data)
    #evaluateRandomly(encoder1, attn_decoder1, test_data)


    #evaluateAndShowAttention("elle a cinq ans de moins que moi .")

    #evaluateAndShowAttention("elle est trop petit .")

    #evaluateAndShowAttention("je ne crains pas de mourir .")

    #evaluateAndShowAttention("c est un jeune directeur plein de talent .")