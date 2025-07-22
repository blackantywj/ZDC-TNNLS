import torch
import torch.nn as nn
import torch.nn.functional as nnf
from typing import Tuple, Optional, List
from copy import deepcopy
from .score_estimator import BertBlock


class Decoder(nn.Module):
    def __init__(self, decoder_config, diffusion_config):
        super().__init__()

        self.num_hidden_layers = decoder_config.num_hidden_layers
        
        arch_config = deepcopy(diffusion_config)
        arch_config.is_conditional = decoder_config.is_conditional
        arch_config.is_decoder = True
        self.blocks = torch.nn.ModuleList(
            [BertBlock(arch_config) for _ in range(0, self.num_hidden_layers)]
        )
        self.fc = nn.Linear(arch_config.hidden_size, arch_config.vocab_size)

        # add IFCap

        # projector for input
        self.linear = nn.Linear(512, 10 * arch_config.hidden_size)
        self.rt_linear = nn.Linear(512, arch_config.hidden_size)

        # learnable prefix embeddings
        self.prefix_const = nn.Parameter(torch.randn(10, arch_config.hidden_size), requires_grad = True)
        self.transformer = Transformer(arch_config.hidden_size, 8, 8)

        ## cross-attention layer
        self.crossatt = att_gt_n_rt(arch_config.hidden_size, 1, 8)


    def forward(self, x, cond_x=None, cond_mask=None):
        extended_cond_mask = self.get_extended_attention_mask(cond_mask)
        for _, block in enumerate(self.blocks):
            x = block(
                hidden_states=x,
                attention_mask=None,
                encoder_hidden_states=cond_x,
                encoder_attention_mask=extended_cond_mask
            )
        x = self.fc(x)
        return x        

    def get_extended_attention_mask(self, attention_mask):
        if attention_mask is None:
            return None
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.iinfo(attention_mask.dtype).min
        return extended_attention_mask
    
class att_gt_n_rt(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int = 8,
        num_heads: int = 8
    ) -> None:
        super(att_gt_n_rt, self).__init__()
        self.transformer = Transformer(d_model, num_layers, num_heads)

    def forward(self, x: torch.Tensor, rtf: torch.Tensor=None) -> torch.Tensor:
        rtf = rtf if rtf is not None else x
        outputs = self.transformer(x, rtf)
        return outputs


class MlpTransformer(nn.Module):

    def __init__(
        self,
        input_size: int,                   # the input size of mlp
        hidden_size: int,                  # the hidden layer size of mlp
        output_size: Optional[int] = None, # the output size of mlp
        act = nnf.relu,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        output_size = output_size if output_size is not None else input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = act
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        query_size: int,
        key_value_size: int,
        num_heads: int,
        bias = True,
        dropout: float = 0.0
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = query_size // num_heads # the size of each head
        self.scale = self.head_size ** -0.5      # normalization factor for each head
        self.to_queries = nn.Linear(query_size, query_size, bias = bias)
        #  projecting key and value together and spliting them for computing efficiently
        self.to_keys_values = nn.Linear(key_value_size, 2 * query_size, bias = bias)
        self.project = nn.Linear(query_size, query_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        key_value = key_value if key_value is not None else query
        b, n, d_query = query.shape
        _, m, _ = key_value.shape
        queries = self.to_queries(query).reshape(b, n, self.num_heads, self.head_size) # (batch_size, n_seq, num_heads, head_size)
        keys_values = self.to_keys_values(key_value).reshape(b, m, 2, self.num_heads, self.head_size) # (batch_size, m_seq, 2, num_heads, head_size)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1] # (batch_size, m_seq, num_heads, head_size), (batch_size, m_seq, num_heads, head_size)
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale # (batch_size, n_seq, m_seq, num_heads)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(dim = 1) # expending dimension, shape: (batch_size, 1, m_seq)
            attention = attention.masked_fill(mask.unsqueeze(dim = 3), float("-inf")) # expending dimension n_seq head and fill -inf according to mask

        attention = attention.softmax(dim = 2) # softmax alongside the dimension of key_value pairs
        outputs = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, d_query) # (batch_size, n_seq, d_query)
        outputs = self.project(outputs)
        return outputs, attention

class TransformerLayer(nn.Module):

    def __init__(
            self,
            query_size: int,
            key_value_size: int,
            num_heads: int,
            mlp_ratio = 4.0,
            bias = False,
            dropout: float = 0.0,
            act = nnf.relu,
            norm_layer: nn.Module = nn.LayerNorm
        ) -> None:
        super(TransformerLayer, self).__init__()
        self.norm1 = norm_layer(query_size)
        self.attn = MultiHeadAttention(query_size, key_value_size, num_heads, bias = bias, dropout = dropout)
        self.norm2 = norm_layer(query_size)
        self.mlp = MlpTransformer(query_size, int(query_size * mlp_ratio), act = act, dropout = dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        query_, self.attention = self.attn(self.norm1(query), key_value, mask)
        query = query + query_
        query = query + self.mlp(self.norm2(query))
        return query

class Transformer(nn.Module):

    def __init__(
            self,
            query_size: int,                      # query size
            num_layers: int,                      # number of layer
            num_heads: int,                       # number of head
            key_value_size: Optional[int] = None, # key/value size
            mlp_ratio: float = 2.0,               # ratio for hidden size in mlp
            act = nnf.relu,                       # activation
            norm_layer: nn.Module = nn.LayerNorm  # normalization
        ) -> None:
        super(Transformer, self).__init__()
        key_value_size = key_value_size if key_value_size is not None else query_size
        layers = []
        for _ in range(num_layers):
            layers.append(TransformerLayer(query_size, key_value_size, num_heads, mlp_ratio = mlp_ratio, act = act, norm_layer = norm_layer))
        self.layers = nn.Sequential(*layers)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        self.attentions = []
        for layer in self.layers:
            query = layer(query, key_value, mask)
            self.attentions.append(layer.attention)
        return query