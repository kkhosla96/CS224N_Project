def pad_terms(terms, pad_token, max_term_length):
    '''
    Note: taken from Kush's implementation in assignment 4 for 224N 
    Pad list of sentences according to the longest sentence in the batch.
    @param sents (List[List[int]]): list of sentences, where each sentence
    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (List[List[int]]): list of sentences where terms shorter
    than the max length sentence are padded out with the pad_token, such that
    each terms in the batch now has equal length.
    '''
    terms_padded = []

    for term in terms:
        diff = max_term_length - len(sentence)
        terms_padded.append((term + [pad_token] * diff)[:max_term_length])

    return terms_padded