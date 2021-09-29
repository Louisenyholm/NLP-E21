from datasets import load_dataset

# read more: https://huggingface.co/datasets/glue
# we will be using sst2 a sentiment dataset by stanford
# compare performance with others:
# https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary

dataset = load_dataset("glue", "sst2")

train = dataset["train"]

print("Examining train set:")
print(train)
print(train.features)
print(train[1]) #sentence number 2, labeled 0 (vs. 1), index 1
# Examining train set:
# Dataset({
#     features: ['sentence', 'label', 'idx'],
#     num_rows: 67349
# })
# {'sentence': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=2, names=['negative', 'positive'], names_file=None, id=None), 'idx': Value(dtype='int32', id=None)}

print("Information about the dataset:")
print(train.info.description)
print("Homepage")
print(train.info.homepage)


print("Examining sentence")
print(type(train["sentence"])) #list of ALL the sentences (below)
#print(train["sentence"])
print(type(train["sentence"][0])) #string
#print(train["sentence"][0]) #hide new secretions from the parental units
print(train[0])


print("Examining label")
print(type(train["label"])) #list of 0s and 1s (below)
#print(train["label"])
print(type(train["label"][0])) #int
#print(train["label"][0])
print(set(train["label"])) # {0, 1} 0 = negative, 1 = positive

print("A few samples:")
for t in range(10):
    sent = train["sentence"][t]
    lab = train["label"][t]
    print(sent, "-", lab)