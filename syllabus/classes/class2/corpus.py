# Import Module
import os

# Read text File  
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        print(f.read())
  
class corpus:
    def __init__(self, path):
        # self.r = realpart
        # self.i = imagpart
    # iterate through all file
    for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path}\{file}"
  
#         # call read text file function
#         read_text_file(file_path)

#create corpus class
# corpus(self, path)
#corpus(train_corpus)


# after tokenisation: list of sentences, of which each is a list of strings (tokens)
#  How to MI
# define a context window/span
# p(w1) = number of appearances/total tokens
# p(w2) = same
# p(w1, w2) = appear together/total tokens
# total tokens = for each list, add length of all together to get the total length
#p(w1, w2) = ...bigrams..
#