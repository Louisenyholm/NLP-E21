import numpy as np
import torch

from datasets import load_dataset
import gensim.downloader as api

from util import batch
from LSTM import RNN
from embedding import gensim_to_torch_embedding

# DATASET ##Loading the dataset "conllpp", subsetting dataset by train.
# Tokens are words. ner_tags are integers reflecting NER class (9 classes in total)
dataset = load_dataset("conllpp")
train = dataset["train"]

# inspect the dataset
print(train["tokens"][:1])
print(train["ner_tags"][:1])
num_classes = train.features["ner_tags"].feature.num_classes
print(num_classes)


<<<<<<< HEAD
# CONVERTING EMBEDDINGS ##getting word embeddings
#We cannot just use gensim, because gensim returns as np embedding, and we want to return it as torch embedding
import numpy as np

import torch

=======
# CONVERTING EMBEDDINGS
>>>>>>> e4732e9599727118ad40b72a55bf0aceda78a541
model = api.load("glove-wiki-gigaword-50")

# convert gensim word embedding to torch word embedding
embedding_layer, vocab = gensim_to_torch_embedding(model)


# PREPARING A BATCH

# shuffle dataset
shuffled_train = dataset["train"].shuffle(seed=1)

# batch it using a utility function (don't spend time on the function, but make sure you understand the output)
batch_size = 10
batches_tokens = batch(shuffled_train["tokens"], batch_size)
batches_tags = batch(shuffled_train["ner_tags"], batch_size)


def tokens_to_idx(tokens, vocab=model.key_to_index):
    """
    Ideas to understand this function:
    - Write documentation for this function including type hints for each arguement and return statement
    - What does the .get method do?
    - Why lowercase? In the dict vocab, it is lowercased (it is made in that way)
    """
    return [vocab.get(t.lower(), vocab["UNK"]) for t in tokens]
    #vocab.get - returns the value for the key (key = t.lower) if it cannot return an idx for the word, then return the idx for the unkown token
    #vocab = dict


<<<<<<< HEAD
# sample batch of 10 sentences
## create batch of 10 sentences (as example)
batch_tokens = train["tokens"][:10]
batch_tags = train["ner_tags"][:10]
batch_tok_idx = [tokens_to_idx(sent) for sent in batch_tokens] #converting sentences to idx
=======
# sample using only the first batch
batch_tokens = next(batches_tokens)
batch_tags = next(batches_tags)
batch_tok_idx = [tokens_to_idx(sent) for sent in batch_tokens]
>>>>>>> e4732e9599727118ad40b72a55bf0aceda78a541
batch_size = len(batch_tokens)

# compute length of longest sentence in batch
##to know how much padding is needed
batch_max_len = max([len(s) for s in batch_tok_idx])

# prepare a numpy array with the data, initializing the data with 'PAD'
# and all labels with -1; initializing labels to -1 differentiates tokens
# with tags from 'PAD' tokens
batch_input = vocab["PAD"] * np.ones((batch_size, batch_max_len)) #batch size down, max length horizontal - filled with the idx of padding token (maybe 400001) = vocab[PAD]
batch_labels = -1 * np.ones((batch_size, batch_max_len)) #batch size vertical, max length horizontal - filled with -1s (same size as above) = the label for our padding token = -1 (which is not a label used, will later be disregarded - in th LSTM loss function RNN)

# copy the data to the numpy array
## for each row...
for i in range(batch_size):
    tok_idx = batch_tok_idx[i] #input the sentence (numbers/idx), and the remainder would be the padding (padding token = 400001)
    tags = batch_tags[i]
    size = len(tok_idx)

    batch_input[i][:size] = tok_idx
    batch_labels[i][:size] = tags


<<<<<<< HEAD
# since all data are indices, we convert them to torch LongTensors
##Longtensor = as interger (same as as.int) = converting it to this format for later use
=======
# since all data are indices, we convert them to torch LongTensors (integers)
>>>>>>> e4732e9599727118ad40b72a55bf0aceda78a541
batch_input, batch_labels = torch.LongTensor(batch_input), torch.LongTensor(
    batch_labels
)

# CREATE MODEL
<<<<<<< HEAD
##Applying the model
from LSTM import RNN

=======
>>>>>>> e4732e9599727118ad40b72a55bf0aceda78a541
model = RNN(
    embedding_layer=embedding_layer, output_dim=num_classes + 1, hidden_dim_size=256
)

# FORWARD PASS
X = batch_input
y = model(X)

loss = model.loss_fn(outputs=y, labels=batch_labels)
# loss.backward()