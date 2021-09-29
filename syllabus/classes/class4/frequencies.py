from collections import Counter


def term_freq(doc) -> dict:
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


def doc_freq(doc_lst) -> dict:
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