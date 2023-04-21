# -*- coding: utf-8 -*-
# @Time    : 2022/4/2 下午5:57
# @Author  : cp
# @File    : layer.py

import math
import torch
import torch.nn as nn




class MultiHeadedAttention(nn.Module):
    """
    Args:
        head_count (int): number of parallel heads
        model_dim (int): the dimension of keys/values/queries,
            must be divisible by head_count
        dropout (float): dropout parameter
    """

    def __init__(self, model_dim, head_count, structure_dim=30, dropout=0.1, use_structure=False, alpha=1.0, beta=1.0):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)

        if use_structure:
            self.linear_structure_k = nn.Linear(structure_dim, self.dim_per_head)
            self.linear_structure_v = nn.Linear(structure_dim, self.dim_per_head)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)
        self.alpha = alpha
        self.beta = beta
        # self.lamb1 = nn.Parameter(torch.Tensor([1]))
        # self.lamb2 = nn.Parameter(torch.Tensor([1]))

    def forward(
        self,
        key,
        value,
        query,
        structure=None,
        mask=None,
        key_padding_mask=None
    ):
        """
        Compute the context vector and the attention vectors.
        Args:
            key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
            value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
            query (`FloatTensor`): set of `query_len`
                    query vectors  `[batch, query_len, dim]`
            structure (`FloatTensor`): set of `query_len`
                    query vectors  `[batch, query_len, query, dim]`

            mask: binary key2key mask indicating which keys have
                    non-zero attention `[batch, key_len, key_len]`
            key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, 1, key_len]`
        Returns:
            (`FloatTensor`, `FloatTensor`) :

            * output context vectors `[batch, query_len, dim]`
            * one of the attention vectors `[batch, query_len, key_len]`
        """

        batch_size,len_k = key.size(0),key.size(1)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)# [b,nhead,seq,head_dim]

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        structure_k = None
        structure_v = None

        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)


        if structure is not None:
            structure_k, structure_v = (
                self.linear_structure_k(structure),
                self.linear_structure_v(structure),
            )

        key = shape(key)  # [batch_size, nhead, seq_len, dim]
        value = shape(value)
        query = shape(query)
        # print('key, query', key.size(), query.size())

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))  # [batch_size, nhead, seq_len, seq_len]
        # print('scores',scores.size())

        if structure_k is not None:  # [batch_size, seq_len, seq_len, dim]
            q = query.transpose(1, 2)  # [batch_size, seq_len, nhead, dim]
            # print(q.size(), structure_k.transpose(2,3).size())
            scores_k = torch.matmul(
                q, structure_k.transpose(2, 3)
            )  # [batch_size, seq_len, nhead, seq_len]
            scores_k = scores_k.transpose(1, 2)  # [batch_size, nhead, seq_len, seq_len]
            # print (scores.size(),scores_k.size())
            scores = scores + self.alpha * scores_k
            # scores = self.lamb1 * scores + self.lamb2 * scores_k

        if key_padding_mask is not None:  # padding mask
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # `[B, 1, 1, seq_len]`
            # print('key_padding_mask', key_padding_mask.size())
            scores = scores.masked_fill(key_padding_mask, -1e18)
        # print('scores_masked', scores)

        if mask is not None:  # key-to-key mask
            mask = mask.unsqueeze(1)  # `[B, 1, seq_len, seq_len]`
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value)

        if structure_v is not None:
            drop_attn_v = drop_attn.transpose(1, 2)
            context_v = torch.matmul(drop_attn_v, structure_v)
            context_v = context_v.transpose(1, 2)
            # print(context.size(),context_v.size())
            context = context + self.beta * context_v

        context = unshape(context)
        output = self.final_linear(context)

        # Return one attn
        top_attn = attn.view(batch_size, head_count, query_len, key_len)[:, 0, :, :].contiguous()

        return output, top_attn







class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                        of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            input: [ batch_size, input_len, model_dim ]
        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

