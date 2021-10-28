import torch
from torch import nn

import numpy as np


def gensim_to_torch_embedding(gensim_wv):
    """
    Input: some kind of model (e.g. Gigaword)
    Converts gensim word embedding to torch word embedding
    Returns dict of vocab and associated torch word embeddings

    - Add type hints on input and output
    - add function description
    - understand the pad and unk embeddings, add an argument which makes these optional. 
        E.g. add_padding = True and add_unknown = True
    """
    embedding_size = gensim_wv.vectors.shape[1]

    # create unknown embedding (for words not seen before - we want to be able to create word embedding for that. We define it arbitrarily as the mean, could also have been all 0s)
    # and padding embedding (in this case: what we use when we want to pad a sentence - here just a vector of 0s)
    unk_emb = np.mean(gensim_wv.vectors, axis=0).reshape((1, embedding_size))
    pad_emb = np.zeros((1, gensim_wv.vectors.shape[1]))

    # add the new embedding #combine the three together into a new matrix
    embeddings = np.vstack([gensim_wv.vectors, unk_emb, pad_emb])

    #converting into torch embedding
    weights = torch.FloatTensor(embeddings)

    #the padding index is the last one (-1)
    emb_layer = nn.Embedding.from_pretrained(embeddings=weights, padding_idx=-1)

    # creating vocabulary #goes from key to index end up with vocab size of 400002, since we added two new tokens (UNK and PAD)
    vocab = gensim_wv.key_to_index
    vocab["UNK"] = weights.shape[0] - 2
    vocab["PAD"] = emb_layer.padding_idx

    return emb_layer, vocab