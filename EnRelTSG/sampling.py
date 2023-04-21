
import random
import torch
import numpy as np
from EnRelTSG import util

def create_train_sample(doc, neg_entity_count: int, neg_rel_count: int, max_span_size: int, rel_type_count: int, over_rate:float):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # all tokens
    token_masks, word2vec_encoding =[],[]
    for t in doc.tokens:
        token_masks.append(create_entity_mask(*t.span, context_size))
        word2vec_encoding.append(t.word2index)

    # positive entities
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes = [], [], [], []
    pos_entity_spans_token, pos_entity_masks_token = [],[]
    for e in doc.entities:
        # print("e.phrase",e.phrase)
        pos_entity_spans.append(e.span)
        pos_entity_spans_token.append(e.span_token)
        pos_entity_types.append(e.entity_type.index)
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))
        pos_entity_masks_token.append(create_entity_mask(*e.span_token, token_count))
        pos_entity_sizes.append(len(e.tokens))

    # positive relations
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks = [], [], [], []
    for rel in doc.relations:
        s1, s2 = rel.head_entity.span_token, rel.tail_entity.span_token
        pos_rels.append((pos_entity_spans_token.index(s1), pos_entity_spans_token.index(s2)))# 
        pos_rel_spans.append((s1, s2))
        pos_rel_types.append(rel.relation_type)
        pos_rel_masks.append(create_rel_mask(s1, s2, token_count))


    # negative entities
    neg_dist_entity_spans, neg_dist_entity_sizes = [], []
    neg_overlap_entity_spans, neg_overlap_entity_sizes = [], []
    neg_dist_entity_spans_token, neg_overlap_entity_spans_token = [],[]
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            span_token = doc.tokens[i:i + size].span_token

            if span not in pos_entity_spans:

                for s1, s2 in pos_entity_spans:
                    if (span[0] >= s1 and span[1] <= s2) or (span[0] <= s1 and span[1] >= s1) or (span[0] <= s2 and span[1] >= s2):
                        neg_overlap_entity_spans.append(span)
                        neg_overlap_entity_spans_token.append(span_token)
                        neg_overlap_entity_sizes.append(size)
                    else:
                        neg_dist_entity_spans.append(span)
                        neg_dist_entity_spans_token.append(span_token)
                        neg_dist_entity_sizes.append(size)

    # count of (inside) overlapping negative mentions and distinct negative mentions
    overlap_neg_count = min(len(neg_overlap_entity_spans), int(neg_entity_count*over_rate))
    dist_neg_count = neg_entity_count - overlap_neg_count

    #overlapping negative span
    neg_overlap_entity_samples = random.sample(list(zip(neg_overlap_entity_spans, neg_overlap_entity_spans_token, neg_overlap_entity_sizes)),
                                                overlap_neg_count)
    neg_overlap_entity_spans, neg_overlap_entity_spans_token, neg_overlap_entity_sizes = zip(*neg_overlap_entity_samples) if neg_overlap_entity_samples \
                                                                else ([], [], [])
    neg_overlap_entity_masks = [create_entity_mask(*span, context_size) for span in neg_overlap_entity_spans]
    neg_overlap_entity_masks_token = [create_entity_mask(*span, token_count) for span in neg_overlap_entity_spans_token]

    # distance negative span
    neg_dist_entity_samples = random.sample(list(zip(neg_dist_entity_spans, neg_dist_entity_spans_token, neg_dist_entity_sizes)),
                                             min(len(neg_dist_entity_spans), dist_neg_count))
    neg_dist_entity_spans, neg_dist_entity_spans_token, neg_dist_entity_sizes = zip(*neg_dist_entity_samples) if neg_dist_entity_samples else \
                                                        ([], [], [])
    neg_dist_entity_masks = [create_entity_mask(*span, context_size) for span in neg_dist_entity_spans]
    neg_dist_entity_masks_token = [create_entity_mask(*span, token_count) for span in neg_dist_entity_spans_token]

    # sample negative terms
    neg_entity_spans_token = list(neg_overlap_entity_spans_token) + list(neg_dist_entity_spans_token)
    neg_entity_sizes = list(neg_overlap_entity_sizes) + list(neg_dist_entity_sizes)
    neg_entity_masks = list(neg_overlap_entity_masks) + list(neg_dist_entity_masks)
    neg_entity_masks_token = list(neg_overlap_entity_masks_token) + list(neg_dist_entity_masks_token)
    neg_entity_types = [0] * len(neg_entity_spans_token)

    # negative relations
    # use only strong negative relations, i.e. pairs of actual (labeled) entities that are not related
    neg_rel_spans = []
    for i1, s1 in enumerate(pos_entity_spans_token):##  pos_entity_spans_token
        for i2, s2 in enumerate(pos_entity_spans_token):
            rev = (s2, s1)
            rev_symmetric = rev in pos_rel_spans and pos_rel_types[pos_rel_spans.index(rev)].symmetric

            # do not add as negative relation sample:
            # neg. relations from an entity to itself
            # entity pairs that are related according to gt
            if s1 != s2 and (s1, s2) not in pos_rel_spans and not rev_symmetric:
                neg_rel_spans.append((s1, s2))

    # sample negative relations
    neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))

    neg_rels = [(pos_entity_spans_token.index(s1), pos_entity_spans_token.index(s2)) for s1, s2 in neg_rel_spans]
    neg_rel_masks = [create_rel_mask(*spans, token_count) for spans in neg_rel_spans]
    neg_rel_types = [0] * len(neg_rel_spans)

    # merge
    # entity
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + neg_entity_sizes
    entity_masks_token = pos_entity_masks_token + neg_entity_masks_token

    # relation
    rels = pos_rels + neg_rels
    rel_types = [r.index for r in pos_rel_types] + neg_rel_types
    rel_masks = pos_rel_masks + neg_rel_masks

    assert len(entity_masks) == len(entity_sizes) == len(entity_types)== len(entity_masks_token)
    assert len(rels) == len(rel_masks) == len(rel_types)

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)
    context_masks = torch.ones(context_size, dtype=torch.bool)
    word2vec_encoding = torch.tensor(word2vec_encoding, dtype=torch.long)

    # masking of tokens
    token_masks = torch.stack(token_masks)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
        entity_masks_token = torch.stack(entity_masks_token)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)
        entity_masks_token = torch.zeros([1, token_count], dtype=torch.bool)

    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        rel_masks = torch.stack(rel_masks)
        rel_types = torch.tensor(rel_types, dtype=torch.long)
        rel_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rels = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1], dtype=torch.long)
        rel_masks = torch.zeros([1, token_count], dtype=torch.bool)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    # relation types to one-hot encoding
    rel_types_onehot = torch.zeros([rel_types.shape[0], rel_type_count], dtype=torch.float32)
    rel_types_onehot.scatter_(1, rel_types.unsqueeze(1), 1)
    rel_types_onehot = rel_types_onehot[:, 1:]  # all zeros for 'none' relation


    adj_matrix, type_matrix = head_to_adj(sent_len=token_count, depHead=doc.dep, edgeType=doc.dep_label_indices)
    adj_graph = torch.tensor(adj_matrix, dtype=torch.long)
    type_graph = torch.tensor(type_matrix, dtype=torch.long)
    pos = torch.tensor(doc.pos_indices, dtype=torch.long)

    return dict(encodings=encodings, context_masks=context_masks,
                token_masks_bool=token_masks_bool, token_masks=token_masks,word2vec_encoding=word2vec_encoding,
                entity_masks=entity_masks, entity_masks_token = entity_masks_token,
                entity_sizes=entity_sizes, entity_types=entity_types,
                rels=rels, rel_masks=rel_masks, rel_types=rel_types_onehot,
                entity_sample_masks=entity_sample_masks, rel_sample_masks=rel_sample_masks,
                adj_graph=adj_graph, type_graph=type_graph,pos=pos)


def create_eval_sample(doc, max_span_size: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # all tokens
    token_masks, word2vec_encoding =[],[]
    for t in doc.tokens:
        token_masks.append(create_entity_mask(*t.span, context_size))
        word2vec_encoding.append(t.word2index)

    # create entity candidates
    entity_spans_token = []
    entity_spans = []
    entity_masks = []
    entity_sizes = []
    entity_masks_token = []

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            span_token = doc.tokens[i:i + size].span_token
            entity_spans.append(span)
            entity_spans_token.append(span_token)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_masks_token.append(create_entity_mask(*span_token, token_count))
            entity_sizes.append(size)

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    context_masks = torch.ones(context_size, dtype=torch.bool)
    word2vec_encoding = torch.tensor(word2vec_encoding, dtype=torch.long)

    # masking of tokens
    token_masks = torch.stack(token_masks)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)

    # entities
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_masks_token = torch.stack(entity_masks_token)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans_token = torch.tensor(entity_spans_token, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_masks_token = torch.zeros([1, token_count], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans_token = torch.zeros([1, 2], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    adj_matrix, type_matrix = head_to_adj(sent_len=token_count, depHead=doc.dep, edgeType=doc.dep_label_indices)
    adj_graph = torch.tensor(adj_matrix, dtype=torch.long)
    type_graph = torch.tensor(type_matrix, dtype=torch.long)
    pos = torch.tensor(doc.pos_indices, dtype=torch.long)

    return dict(encodings=encodings, context_masks=context_masks,
                token_masks_bool=token_masks_bool, token_masks=token_masks,word2vec_encoding=word2vec_encoding,
                entity_masks=entity_masks, entity_spans_token=entity_spans_token, entity_masks_token= entity_masks_token,
                entity_sizes=entity_sizes, entity_sample_masks=entity_sample_masks,
                adj_graph=adj_graph, type_graph=type_graph,pos=pos,entity_spans=entity_spans)



def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch


def head_to_adj(sent_len, depHead, edgeType, directed=False, self_loop=True):
    """
    Convert a sequence of head indexes into a 0/1 matrix and edge type matrix.
    Note : sent_len is org sentence len
    """
    adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)
    type_matrix = np.zeros((sent_len, sent_len), dtype=np.int64)


    for idx, head in enumerate(depHead):

        adj_matrix[idx, int(head) - 1] = 1
        type_matrix[idx, int(head) - 1] = edgeType[idx]

        if not directed:
            adj_matrix[int(head) - 1, idx] = 1
            type_matrix[int(head) - 1, idx] = edgeType[idx]
        if self_loop:
            adj_matrix[idx, idx] = 1
            type_matrix[idx, idx] = 2 # <self> edge type id

    return adj_matrix, type_matrix




