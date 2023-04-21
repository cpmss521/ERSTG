# -*- coding: utf-8 -*-
# @Time    : 2022/5/1 下午7:09
# @Author  : cp
# @File    : Encoder.py

import torch
import torch.nn as nn
from EnRelTSG.layer import PositionwiseFeedForward, MultiHeadedAttention


class TypeGATLayer(nn.Module):
    def __init__(
        self, d_model, heads, d_ff, dropout, att_drop=0.1, use_structure=True, dep_dim=40, alpha=1.0, beta=1.0
    ):
        super(TypeGATLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            d_model, heads, structure_dim=dep_dim, dropout=att_drop, use_structure=use_structure,
            alpha=alpha, beta=beta)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask=None, key_padding_mask=None, structure=None):
        """
    Args:
       input (`FloatTensor`): h_token, `[batch, seq_len, H]`
       mask: adj_graph, binary key2key mask indicating which keys have
             non-zero attention `[batch, seq_len, seq_len]`
       key_padding_mask: binary padding mask indicating which keys have
             non-zero attention `[batch, 1, seq_len]`
       structure: dep_type embedding
    return:
       res:  [batch, seq_len, H]
    """

        # input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(
            inputs,
            inputs,
            inputs,
            mask=mask,
            key_padding_mask=key_padding_mask,
            structure=structure,
        )
        out = self.dropout(context) + inputs
        out = self.feed_forward(out)

        return out


class nTypeGAT(nn.Module):

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        dropout,
        att_drop=0.1,
        use_structure=True,
        dep_dim=40,
        alpha=1.0,
        beta=1.0,
    ):
        super(nTypeGAT, self).__init__()

        self.num_layers = num_layers
        self.transformer = nn.ModuleList(
            [
                TypeGATLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    att_drop=att_drop,
                    use_structure=use_structure,
                    dep_dim=dep_dim,
                    alpha=alpha,
                    beta=beta,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, src, mask=None, src_key_padding_mask=None, structure=None):
        """
    Args:
       src (`FloatTensor`): h_token `[batch, seq_len, H]`
       mask: adj_graph, binary key2key mask indicating which keys have
             non-zero attention `[batch, seq_len, seq_len]`
       src_key_padding_mask: binary key padding mask indicating which keys have
             non-zero attention `[batch, 1, seq_len]`
       structure: dep_type embedding  
    return:
       out_trans (`FloatTensor`): `[batch, seq_len, H]`
    """

        out = src  # [B, seq_len, H]

        # Run the forward pass of every layer of the transformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask, src_key_padding_mask, structure=structure)
        out = self.layer_norm(out)  # [B, seq, H]
        return out




class TSGEncoder(nn.Module):
    def __init__(self, opt,device):
        super(TSGEncoder, self).__init__()
        self.opt = opt
        self._device = device
        self.tgat = nTypeGAT(opt.num_layers,opt.d_model,opt.heads,opt.bert_dim,
                             opt.dropout, opt.att_drop,dep_dim=opt.dep_dim,alpha=opt.alpha,beta=opt.beta)

        self.fc = nn.Linear(opt.bert_dim*2 + opt.pos_dim, opt.bert_dim)
        self.output_dropout = nn.Dropout(opt.output_dropout)

    def forward(self, h_token, adj_graph, depType_embed, lengths, pos_embed):

        key_padding_mask = self.sequence_mask(lengths) if lengths is not None else None  # [B, seq_len]

        tgat_output = self.tgat(h_token, adj_graph.eq(0), key_padding_mask, depType_embed)# [B, seq_len, H]

        pos_output = self.local_attn(h_token, pos_embed, self.opt.num_layer, self.opt.w_size)#[B,seq_len,pos_dim]

        output = torch.cat((tgat_output, pos_output, h_token), dim=-1)
        output = self.fc(output)
        output = self.output_dropout(output)#[B,seq_len,H]

        return output


    def local_attn(self, x, pos_embed, num_layer, w_size):
        """
        :param x:
        :param pos_embed:
        :return:[batch size, seq_len, pos_dim]
        """
        batch_size, seq_len, feat_dim = x.shape
        pos_dim = pos_embed.size(-1)
        output = pos_embed
        for i in range(num_layer):
            val_sum = torch.cat([x, output], dim=-1)  # [batch size, seq_len, feat_dim+pos_dim]
            attn = torch.matmul(val_sum, val_sum.transpose(1, 2))  # [batch size, seq_len, seq_len]
            # pad size = seq_len + (window_size - 1) // 2 * 2
            pad_size = seq_len + w_size * 2
            mask = torch.zeros((batch_size, seq_len, pad_size), dtype=torch.float).to(
                device=self._device)
            for i in range(seq_len):
                mask[:, i, i:i + w_size] = 1.0
            pad_attn = torch.full((batch_size, seq_len, w_size), -1e18).to(
                device=self._device)
            attn = torch.cat([pad_attn, attn, pad_attn], dim=-1)

            local_attn = torch.softmax(torch.mul(attn, mask), dim=-1)#B,S,pad_size
            local_attn = local_attn[:, :, w_size:pad_size - w_size]  # [batch size, seq_len, seq_len]
            local_attn = local_attn.unsqueeze(dim=3).repeat(1, 1, 1, pos_dim)
            output = output.unsqueeze(dim=2).repeat(1, 1, seq_len, 1)
            output = torch.sum(torch.mul(output, local_attn), dim=2)  # [batch size, seq_len, pos_dim]
        return output


    def sequence_mask(self, lengths, max_len=None):
        """
        create a boolean mask from sequence length `[batch_size, 1, seq_len]`
        """

        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return torch.arange(0, max_len, device=self._device).type_as(lengths).unsqueeze(0).expand(
            batch_size, max_len
        ) >= (lengths.unsqueeze(1))