import torch
import numpy as np
from torch import nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from EnRelTSG import sampling
from EnRelTSG import util
from EnRelTSG import Encoder,layer



def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h


def get_head_tail_rep(h, entity_masks):
    """
    :param h: torch.tensor [batch size, seq_len, feat_dim]
    :param entity_masks:
    :return:
    """
    m = entity_masks.to(dtype=torch.long)
    k = torch.tensor(np.arange(0, entity_masks.size(-1)), dtype=torch.long)
    k = k.unsqueeze(0).unsqueeze(0).repeat(entity_masks.size(0), entity_masks.size(1), 1).to(m.device)
    mk = torch.mul(m, k)  # element-wise multiply
    mk_max = torch.argmax(mk, dim=-1, keepdim=True)
    mk_min = torch.argmin(mk, dim=-1, keepdim=True)
    head_tail_index = torch.cat([mk_min, mk_max], dim=-1) # [batch size, entity_num, 2]

    res = []
    batch_size = head_tail_index.size()[0]
    entity_num = head_tail_index.size()[1]
    for b in range(batch_size):
        temp = []
        for t in range(entity_num):
            temp.append(torch.index_select(h[b], 0, head_tail_index[b][t]).view(-1))
        res.append(torch.stack(temp, dim=0))
    res = torch.stack(res)
    return res


class EnRelTSG(BertPreTrainedModel):
    """ TypeGAT-based model to jointly extract entities and relations """

    VERSION = '1.1'

    def __init__(self, config: BertConfig, embed: torch.tensor, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, bert_dropout:float, word_dropout:float,pos_num:int, pos_dim:int,
                 dep_num:int,dep_dim:int,freeze_transformer: bool, args, device,lstm_drop:float=0.5, lstm_layers:int=1,
                 max_pairs:int=100,pool_type:str = "max", use_glove: bool = True, use_lstm:bool = False):
        super(EnRelTSG, self).__init__(config)

        # BERT model
        self.bert = BertModel(config)
        self._device = device
        #Encoder
        self.TSG = Encoder.TSGEncoder(opt=args,device=self._device)
        # decoder
        # self.atten = layer.MultiHeadAttention(config.hidden_size, n_head=4, d_k=64, d_v=64, dropout=0.1,flag=True)

        # layers
        self.rel_classifier = nn.Linear(config.hidden_size * 3 + size_embedding * 2, relation_types)
        self.entity_classifier = nn.Linear(config.hidden_size * 4 + size_embedding, entity_types)
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(prop_drop)
        self.bert_dropout = nn.Dropout(bert_dropout)
        self.word_drop = nn.Dropout(word_dropout)


        if embed is not None:
            self.wordvec_size = embed.size(-1)
        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs
        self.pool_type = pool_type
        self.use_glove = use_glove
        self.use_lstm = use_lstm

        lstm_hidden = config.hidden_size

        if self.use_glove:
            lstm_hidden += self.wordvec_size
        self.pod_embedding = nn.Embedding(pos_num, pos_dim, padding_idx=0)
        self.dep_embedding = nn.Embedding(dep_num, dep_dim, padding_idx=0)

        if not self.use_lstm and self.use_glove:
            self.reduce_dimension = nn.Linear(lstm_hidden, config.hidden_size)

        if self.use_lstm:
            self.lstm = nn.LSTM(input_size = lstm_hidden, hidden_size = config.hidden_size//2,
                                num_layers = lstm_layers,  bidirectional = True, dropout = lstm_drop,
                                batch_first = True)

        # weight initialization
        self.init_weights()

        if self.use_glove:
            self.word2vec_embedding = nn.Embedding.from_pretrained(embed, freeze=False)

        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    def _common_forward(self, encodings: torch.tensor, context_masks: torch.tensor,
                       token_masks: torch.tensor, token_masks_bool: torch.tensor,word2vec_encoding: torch.tensor,
                        adj_graph: torch.tensor, type_graph: torch.tensor,pos: torch.tensor):

        # get contextualized token embeddings from last transformer layer
        sequence_output = self.bert(input_ids=encodings,attention_mask =context_masks.float())[0]
        sequence_output = self.bert_dropout(sequence_output)#(B #S H)
        h_token = self.combine(sequence_output, token_masks, self.pool_type)#(B,S,H)

        h_token = self.add_extra_embedding(h_token, word2vec_encoding)

        token_count = token_masks_bool.long().sum(-1, keepdim=True)
        sentence_lengths = token_count.squeeze(-1).cpu().tolist()
        if self.use_lstm:
            h_token = nn.utils.rnn.pack_padded_sequence(input = h_token, lengths = sentence_lengths,
                                                        enforce_sorted = False, batch_first = True)
            h_token, (_, _) = self.lstm(h_token)
            h_token, _ = nn.utils.rnn.pad_packed_sequence(h_token, batch_first=True)
        #h_token:(B,S,H)
        pos_embed = self.pod_embedding(pos)
        depType_embed = self.dep_embedding(type_graph)
        #h_out:(B,S,H)
        h_out = self.TSG(h_token=h_token, adj_graph=adj_graph, depType_embed=depType_embed,
                                 lengths=token_count.squeeze(-1), pos_embed=pos_embed)


        return sequence_output, h_out

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor,
                       token_masks_bool: torch.tensor, token_masks: torch.tensor, word2vec_encoding: torch.tensor,
                       entity_masks: torch.tensor, entity_masks_token:torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor,
                       adj_graph: torch.tensor, type_graph: torch.tensor, pos: torch.tensor):


        sequence_output, h_out = self._common_forward(encodings,context_masks,token_masks,token_masks_bool,
                                            word2vec_encoding,adj_graph,type_graph,pos)

        batch_size = encodings.shape[0]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, sequence_output, h_out, entity_masks,
                                                                entity_masks_token, size_embeddings)

        # classify relations
        h_large = h_out.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool,
                                                        size_embeddings,
                                                        relations, rel_masks,
                                                        h_large, i,
                                                        )
            # apply sigmoid
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits

        return entity_clf, rel_clf

    def add_extra_embedding(self, h_token, word2vec_encoding):
        embeds = [h_token]

        if self.use_glove:
            word_embed = self.word2vec_embedding(word2vec_encoding)
            word_embed = self.word_drop(word_embed)
            embeds.append(word_embed)

        h_token = torch.cat(embeds, dim=-1)
        if len(embeds) > 1 and not self.use_lstm:
            h_token = self.reduce_dimension(h_token)
        return h_token

    def combine(self, sub, sup_mask, pool_type = "max" ):
        sup = None
        if len(sub.shape) == len(sup_mask.shape) :
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        else:
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        return sup

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor,
                           token_masks_bool: torch.tensor, token_masks: torch.tensor, word2vec_encoding: torch.tensor,
                           entity_masks: torch.tensor, entity_spans_token: torch.tensor, entity_masks_token: torch.tensor,
                           entity_sizes: torch.tensor,  entity_sample_masks: torch.tensor,
                           adj_graph: torch.tensor, type_graph: torch.tensor, pos: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        sequence_output, h_out = self._common_forward(encodings, context_masks, token_masks, token_masks_bool,
                                             word2vec_encoding, adj_graph, type_graph, pos)

        batch_size = encodings.shape[0]
        ctx_size = token_masks_bool.shape[-1]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, sequence_output, h_out, entity_masks,
                                                                entity_masks_token, size_embeddings)

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf,
                                                                    entity_spans_token,
                                                                    entity_sample_masks,
                                                                    ctx_size,
                                                                    )

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h_out.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool,
                                                         size_embeddings,
                                                         relations, rel_masks,
                                                         h_large, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations

    def _classify_entities(self, encodings, sequence_output, h_out, entity_masks, entity_masks_token, size_embeddings):
        # max pool entity candidate spans
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + sequence_output.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        m_token = (entity_masks_token.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool_token = m_token + h_out.unsqueeze(1).repeat(1, entity_masks_token.shape[1], 1, 1)
        entity_spans_pool = entity_spans_pool_token.max(dim=2)[0] + entity_spans_pool

        # get cls token as candidate context representation,(B,S,H)
        entity_ctx = get_token(sequence_output, encodings, self._cls_token)

        # get head and tail token representation
        head_tail_rep = get_head_tail_rep(sequence_output, entity_masks)  # [batch size, entity_num, bert_dim*2)
        head_tail_rep = get_head_tail_rep(h_out, entity_masks_token) + head_tail_rep  # [batch size, entity_num, bert_dim*2)

        # create candidate representations including context, max pooled span and size embedding
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings, head_tail_rep], dim=2)
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool


    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        batch_size = relations.shape[0]

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]


        # get pairs of entity candidate representations
        entity_pairs = util.batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)# [1, 1, 51, 1]
        #h: [1, 1, 35, 768]
        rel_ctx = m + h
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)
        # classify relation candidates
        chunk_rel_logits = self.rel_classifier(rel_repr)

        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans_token, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans_token[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, inference=False, **kwargs):
        if not inference:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_inference(*args, **kwargs)


# Model access

_MODELS = {
    'EnRelTSG': EnRelTSG,
}


def get_model(name):
    return _MODELS[name]
