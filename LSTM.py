import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple


HIDDEN_SIZE = 256
EMBED_SIZE = 256

class LSTM(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE, bidirectional=False, vocab=None, embeddings=None):
        super(LSTM, self).__init__()
        self.embeddings = embeddings
        self.embed_size = embeddings.embedding_dim
        self.LSTM = nn.LSTM(self.embed_size, hidden_size, bidirectional=bidirectional)
        self.vocab = vocab
        self.linear = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)
    
    def forward(self, candidates: List[List[str]]) -> torch.Tensor:
        """ Takes a mini-batch of candidates and computes the log-likelihood that
        they are glossary terms
        
        @param candidates (List[List[str]]): Batch of candidates (need to be padded)
        
        @returns scores (Tensor): a tensor of shape (batch_size, ) representing the
        log-likelihood that a candidate is a glossary term
        """

        candidate_lengths = sorted([min(len(candidate), self.vocab.get_term_length()) for candidate in candidates], reverse = True)
        candidates = sorted(candidates, key= lambda candidate : len(candidate), reverse=True)
        candidates_padded = self.vocab.to_input_tensor(candidates).permute(1, 0)   # Tensor: (max_length, batch_size))
        enc_hiddens = self.encode(candidates_padded, candidate_lengths)

        # this code taken from https://blog.nelsonliu.me/2018/01/24/extracting-last-timestep-outputs-from-pytorch-rnns/
        idx = (torch.LongTensor(candidate_lengths) - 1).view(-1, 1).expand(len(candidate_lengths), enc_hiddens.size(2))
        idx = idx.unsqueeze(1)
        last_hiddens = enc_hiddens.gather(1, idx).squeeze(1)
        probs = torch.sigmoid(self.linear(last_hiddens))
        return probs

    def encode(self, candidates_padded: torch.Tensor, candidate_lengths: List[int]) -> torch.Tensor:
        """ Apply the encoder to candidates to obtain encoder hidden states.

        @param source_padded (Tensor): Tensor of padded candidates with shape (max_length, batch_size), where
                                        b = batch_size, max_length = maximum source sentence length. Note that 
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h or 2h), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
                                        If the LSTM is bidirectional, then the final dimension is 2h
        """
        enc_hiddens, dec_init_state = None, None
        X = self.embeddings(candidates_padded)
        X = nn.utils.rnn.pack_padded_sequence(X, candidate_lengths)
        enc_hiddens, _ = self.LSTM(X)
        enc_hiddens, _ = nn.utils.rnn.pad_packed_sequence(enc_hiddens, batch_first=True)
        return enc_hiddens
