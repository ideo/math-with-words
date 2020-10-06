from copy import copy
from string import punctuation

import spacy  


def load_spacy():
    nlp = spacy.load("en_core_web_sm")

    # Remove certain stop words
    words_to_keep = ["no", "not", "none", "out"]
    for wrd in words_to_keep:
        nlp.vocab[wrd].is_stop = False

    return nlp
NLP = load_spacy()


def nlp_pipeline(df, nlp):
    clean(df["raw"])
    df["processed"] = df["raw"].apply(lambda doc: tokenize(doc, nlp))
    return df


# Clean
def clean(raw_column):
    deepL_phrase = "(after the -te form of a verb) not ..."
    raw_column.replace(to_replace=deepL_phrase, value="", inplace=True)
    

# Tokenize
    # Remove stopwords
    # Lemmatize

def tokenize(doc, nlp):
    doc = remove_stop_words(doc, nlp)
    doc = lemmatize(doc, nlp)
    return doc


def remove_stop_words(doc, nlp):
    tokens = doc.split(" ")
    tokens = [tkn.strip() for tkn in tokens]
    tokens = [tkn for tkn in tokens if not nlp.vocab[tkn].is_stop]
    doc = " ".join(tokens)
    return doc


def lemmatize(doc, nlp):
    doc = nlp(doc)
    tokens = [word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in doc]
    tokens = [tkn for tkn in tokens if tkn not in punctuation]
    doc = " ".join(tokens)
    return doc


# N-Gram Demo
def ngram_demo(test_sentence, ngram_range):
    tokens = test_sentence.split(" ")
    tokens = [t.strip().lower().replace(".", "") for t in tokens]
    output = copy(tokens)

    if ngram_range > 1:
        for N in range(2, ngram_range+1):
            for ii in range(len(tokens) - N+1):
                output.append(" ".join(tokens[ii:ii+N]))

    return output


# Topic Modeling
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def retrieve_topic_keywords(model, feature_names, n_top_words):
    topic_keywords = {}
    for topic_idx, topic in enumerate(model.components_):
        # message = "Topic #%d: " % topic_idx
        # message += " ".join([feature_names[i]
        #                      for i in topic.argsort()[:-n_top_words - 1:-1]])
        # print(message)
        topic_keywords[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return topic_keywords
