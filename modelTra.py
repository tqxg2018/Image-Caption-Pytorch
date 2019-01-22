#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torchvision
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        features = self.resnet(images) # 64 * 2048
        features = features.reshape(features.size(0), -1) # 64 * 2048
        features = self.bn(self.linear(features)) # 64 * 512
        return features
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout = 0.5): #512 512 9490 1 
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.decode_step = nn.LSTMCell(embed_size, hidden_size, bias=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, images, captions, length):
        
        batch_size = images.size(0) #64
        vocab_size = self.vocab_size #vocab size
        
        caption_lengths, sort_ind = length.squeeze(1).sort(dim=0, descending=True)
        features = images[sort_ind] #2048
        captions = captions[sort_ind] #512
        
        embeddings = self.embedding(captions)
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        h, c = self.decode_step(features)  # (batch_size_t, decoder_dim)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h, c = self.decode_step(embeddings[:batch_size_t, t, :], (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
        
        return predictions, captions, decode_lengths, sort_ind
