from math import ceil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def fix_torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRecBackBone(nn.Module):
    def __init__(self, item_num, config):
        super(SASRecBackBone, self).__init__()
        self.item_num = item_num
        self.pad_token = item_num

        self.item_emb = nn.Embedding(self.item_num+1, config['hidden_units'], padding_idx=self.pad_token)
        self.pos_emb = nn.Embedding(config['maxlen'], config['hidden_units'])
        self.emb_dropout = nn.Dropout(p=config['dropout_rate'])

        self.attention_layernorms = nn.ModuleList() # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        self.last_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-8)

        for _ in range(config['num_blocks']):
            new_attn_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer =  nn.MultiheadAttention(
                config['hidden_units'],config['num_heads'],config['dropout_rate']
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(config['hidden_units'], eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(config['hidden_units'], config['dropout_rate'])
            self.forward_layers.append(new_fwd_layer)

        fix_torch_seed(config['manual_seed'])
        self.initialize()

    def initialize(self):
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_uniform_(param.data)
            except:
                pass # just ignore those failed init layers

    def log2feats(self, log_seqs):
        device = log_seqs.device
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.arange(log_seqs.shape[1]), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs == self.pad_token
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.full((tl, tl), True, device=device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )

            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs)

        log_feats = log_feats.view(-1, log_feats.shape[-1])
        pos_seqs = pos_seqs.view(-1)
        neg_seqs =  neg_seqs.permute(0, 2, 1).reshape(-1, neg_seqs.shape[1]) # (bs * seq, neg)

        pos_embs = self.item_emb(pos_seqs) # (bs * seq, hd)
        neg_embs = self.item_emb(neg_seqs) # (bs * seq, neg, hd)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats[:, None, :] * neg_embs).sum(dim=-1) # (bs * seq, neg)

        return pos_seqs, pos_logits, neg_logits

    def score(self, seq):
        '''
        Takes 1d sequence as input and returns prediction scores.
        '''
        maxlen = self.pos_emb.num_embeddings
        log_seqs = torch.full([maxlen], self.pad_token, dtype=torch.int64, device=seq.device)
        log_seqs[-len(seq):] = seq[-maxlen:]
        log_feats = self.log2feats(log_seqs.unsqueeze(0))
        final_feat = log_feats[:, -1, :] # only use last QKV classifier

        item_embs = self.item_emb.weight
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
    
    def score_2d(self, seqs, device):
        maxlen = max([len(i) for i in seqs])
        seqs_padded = [
            np.pad(
                s, (maxlen - len(s), 0),
                mode='constant',
                constant_values=self.pad_token
            ) for s in seqs
        ]
        seqs_padded = torch.tensor(
            seqs_padded, device=device, dtype=torch.long
        )[:, -self.pos_emb.num_embeddings:]

        final_feat = self.log2feats(seqs_padded)[:, -1, :]

        item_embs = self.item_emb.weight
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits

    def score_with_state(self, seq):
        '''
        Takes 1d sequence as input and returns prediction scores.
        '''
        maxlen = self.pos_emb.num_embeddings
        log_seqs = torch.full([maxlen], self.pad_token, dtype=torch.int64, device=seq.device)
        log_seqs[-len(seq):] = seq[-maxlen:]
        log_feats = self.log2feats(log_seqs.unsqueeze(0))
        final_feat = log_feats[:, -1, :] # only use last QKV classifier

        item_embs = self.item_emb.weight
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits, final_feat[0]


class SASRec(SASRecBackBone):
    def __init__(self, item_num, config):
        super().__init__(item_num, config)

        self.fwd_type = config['fwd_type']

        if self.fwd_type in ['bce', 'gbce']:
            self.n_neg_samples = config['n_neg_samples']

        if self.fwd_type == 'rece':
            self.n_buckets = config['n_buckets']
            self.buckets_per_chunk = config['buckets_per_chunk']
            self.n_extra_chunks = config['n_extra_chunks']
            self.rounds = config['rounds']

            padded_ds = ceil(self.item_emb.weight.shape[0] / self.n_buckets) * self.n_buckets

            self.item_emb = nn.Embedding(padded_ds, config['hidden_units'], padding_idx=self.pad_token)
            torch.nn.init.xavier_uniform_(self.item_emb.weight.data)

            with torch.no_grad():
                self.item_emb.weight[self.pad_token+1:, :] = 0.
            
        elif self.fwd_type == 'gbce':
            alpha = self.n_neg_samples / (item_num - 1.)
            self.beta = alpha * (config['gbce_t'] * (1. - 1. / alpha) + 1. / alpha)

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        if self.fwd_type == 'rece':
            return self.rece_forward(log_seqs, pos_seqs)

        elif self.fwd_type == 'bce':
            return self.bce_forward(log_seqs, pos_seqs, neg_seqs)
        
        elif self.fwd_type == 'gbce':
            return self.gbce_forward(log_seqs, pos_seqs, neg_seqs)

        elif self.fwd_type == 'ce':
            return self.ce_forward(log_seqs, pos_seqs)
        
        elif self.fwd_type == 'dross':
            return self.dross_forward(log_seqs, pos_seqs, neg_seqs)

        else:
            raise ValueError(f'Wrong fwd_type type - {self.fwd_type}')

    def bce_forward(self, log_seqs, pos_seqs, neg_seqs):
        device = log_seqs.device
        pos_seqs, pos_logits, neg_logits = super().forward(log_seqs, pos_seqs, neg_seqs)

        pos_logits = pos_logits[:, None]

        pos_labels = torch.ones(pos_logits.shape, device=device)
        neg_labels = torch.zeros(neg_logits.shape, device=device)

        logits = torch.cat([pos_logits, neg_logits], -1)

        gt = torch.cat([pos_labels, neg_labels], -1)

        mask = (pos_seqs != self.pad_token).float()

        loss_per_element = \
            torch.nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none').mean(-1) * mask
        loss = loss_per_element.sum() / mask.sum()
        
        return loss

    def ce_forward(self, log_seqs, pos_seqs):
        emb = self.log2feats(log_seqs)

        logits = emb @ self.item_emb.weight.T
        indices = torch.where(pos_seqs.view(-1) != self.pad_token)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1])[indices], pos_seqs.view(-1)[indices], reduction='mean')
        return loss