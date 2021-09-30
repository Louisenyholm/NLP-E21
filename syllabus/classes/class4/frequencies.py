from collections import Counter

# Term frequency
def term_freq(tokens) -> dict:
    """
    Takes in a list of tokens (str) and return a dictionary of term frequency of each token
    (doc = a list of tokens)
    """
    #counting occurrences of each unique token
    term_count = Counter(tokens)
    #total number of tokens
    n_tokens = len(tokens)

    #put into dictionary in the formant token
    tf_dict = {token: count/n_tokens for token, count in term_count.items()}
    
    #return tf_dict
    return tf_dict

# Document frequency where DF is the number of documents in which the word is present
#If one or more occurrence(s) in the document/list counts as 1 occurence (not more)
def doc_freq2(doc_lst) -> dict:
    """
    Takes in a list of documents which each is a list of tokens (str) and return a dictionary of frequencies for each token over all the documents. E.g. {"Aarhus": 20, "the": 2301, ...}
    """
    #empty list
    all_counters_lst = []

    #iterating through docs, first keeping only unique elements in the lists, then using Counter
    for doc in doc_lst:
        #unique elements in list
        doc = set(doc)
        #append to list a counter with frequencies in each doc
        all_counters_lst.append(Counter(doc))

    #Empty counter (.update works on counters, not lists)
    all_counters = Counter()

    #iterating thorugh counters, updating (=adding)
    for counter in all_counters_lst:
        all_counters.update(counter)

    return dict(all_counters)


# Document frequency
def doc_freq(doc_lst) -> dict:
    """
    Takes in a list of documents which each is a list of tokens (str) and return a dictionary of frequencies for each token over all the documents. E.g. {"Aarhus": 20, "the": 2301, ...}
    """
    #empty list (can only append to lists, not to counters)
    all_counters_lst = []

    #Iterating through docs
    for doc in doc_lst:
        #append to list a counter with frequencies in each doc
        all_counters_lst.append(Counter(doc))

    #Empty counter (.update works on counters, not lists)
    all_counters = Counter()

    #iterating thorugh counters, updating (=adding)
    for counter in all_counters_lst:
        all_counters.update(counter)

    return dict(all_counters)




# Misunderstood
def misunderstood_token_freq(doc) -> dict:
    """
    Takes in a list of tokens (str) and return a dictionary of term frequency of each token
    (doc = a list of tokens)
    """
    
    #counts number of cases of the different class types (pos)
    pos_counts = Counter([token.pos_ for token in doc])

    # For class type tag (pos) and count (both from above),
    # take the pos tag and the count divided by length to get freq
    tf_lst = [(pos, count/len(doc)) for pos, count in pos_counts.items()]
    #convert the list of tuples to a dictionary
    tf_dict = dict(tf_lst)

    return tf_dict


def misunderstood_doc_freq(doc_lst) -> dict:
    """
    Takes in a list of documents which each is a list of tokens (str) and return a dictionary of frequencies for each token over all the documents. E.g. {"Aarhus": 20, "the": 2301, ...}
    """
    #empty list (.append works on lists, not counters)
    all_counters_lst = []

    #Iterating through docs
    for doc in doc_lst:
        #append to list a counter with frequencies of pos in each doc
        all_counters_lst.append(Counter([token.pos_ for token in doc]))

    #Empty counter (.update works on counters, not lists)
    all_counters = Counter()

    #iterating thorugh counters, updating (=adding)
    for counter in all_counters_lst:
        all_counters.update(counter)

    return all_counters