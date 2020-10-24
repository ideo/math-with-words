import pickle
from copy import copy
from collections import defaultdict

import pandas as pd
import streamlit as st
# from spacy.lang.en.stop_words import STOP_WORDS
# from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from num2words import num2words

# from aws import get_s3_bucket, load_pickled_dataframe, load_pickled_object
# from nlp_pipeline import NLP, load_spacy, nlp_pipeline, ngram_demo, retrieve_topic_keywords
from nlp_pipeline import ngram_demo, retrieve_topic_keywords


# Helper Functions
def rec_dd():
    return defaultdict(rec_dd)


def load_pickled_object(filename):
    obj = pickle.load(open(filename, "rb"))
    if isinstance(obj, pd.DataFrame):
        obj.drop(columns=["Unnamed: 0"], inplace=True)
    return obj


st.title("Project Enlighten Survey Analysis")
st.header("An Introduction to Topic Modeling")
opening_msg = """
Hi Koko! I made this so we could explore the survey results together. Your question 
for me regarding the free response quesions was, *"What is the typical response?"* Here's 
how we'll try to answer that:  
    
1. First, we'll clean up the text to make it easier to analyze.  
2. Next, we'll count all the words in each response.  
3. Then, we'll group words that commonly appear together. We'll call these groups topics.  
4. Lastly, we'll count how many responses belong to each topic. If one outnumbers all the 
others, there's our most typical response!

You'll see along the way that when doing math with words there often isn't a mathematical 
way to say we're doing things the best way, and we need to assess for ourselves how useful 
or reasonable the results are. We'll also see that each changes to each step can affect 
the outcome, so we'll probably take an iterative approach to finding these topics. Let's 
get to it!
"""
st.write(opening_msg)

# Translation
st.header("Step 1: Translation")
msg = """
ごめんなさい、でも日本語わからない。I can't do NLP in a language I don't know. So how does the 
translation look? Nothing I can really do here. All hail DeepL.
"""
st.write(msg)

# bucket = get_s3_bucket()
# df = load_pickled_dataframe(bucket, "attitudes_survey_translation_9_25.pkl")
df = load_pickled_object("attitudes_survey_translation_9_25.pkl")
st.dataframe(df)
st.write(f"Summary: {df.shape[0]} responses to {df.shape[1]-1} questions.")

st.header("Step 2: Lemmatization")
msg = """
We'll basically be counting words here. But computers are dumb. The words `run`, `running`, and `ran` 
will be counted as differnt words. Through a process called lemmatization, we reduce all 
words to their _lemma_, such as `run`. This will make it easier to directly compare responses. Additionally 
we'll perform some basic cleaning like breaking up contractions and removing punctuation.

This process takes a fair amount of time to run, so I've done it ahead of time and saved the results. 
Here's a snippet:
"""
st.write(msg)

free_response_questions = {
    "holding_back": "If you want to contribute more to the environment, I'd like to ask you -- what do you feel is currently holding you back from taking action to contribute more?",
    "habits": "Are there any actions or habits that you have devised that are unusual for those around you that lead to eco/environmental issues?",
    }
chosen_question = st.radio(label="Free Response Question to Analyze", options=list(free_response_questions.values()))
reverse_col_to_key = {v:k for k,v in free_response_questions.items()}
question = reverse_col_to_key[chosen_question]

analysis_dfs = load_pickled_object("analysis_dfs.pkl")
analysis_df = analysis_dfs[question]

st.write("First 10 Responses")
st.table(analysis_df[["raw", "processed"]].head(10))#.set_index("original_japanese"))


st.header("Step 3: Vectorization")
msg = """
Computers do math with numbers and not words. Here is where we count the words. Two important concepts here: 
what we're counting and how we're counting them.
"""
st.write(msg)

st.subheader("What Not to Count: Stop Words")
msg = """
We'll make sure not to count common, meaningless words such as `the` and `in` called *stop words*. Otherwise, 
they will be counted most often and our topics will be dominated by meaningless words.

The library I'm using includes a helpful stop words list, but it's not one-size-fits-all. I'm going to modify it 
a bit. I'm going to remove negative words like, like `no`, `not` and `none` from the list. I'm also going to remove 
the word `out` from the list, because I noticed that by including it the response, "I am out of money",  turned 
into, "I am money". Slightly different meaning. If you notice other words that you wish to keep, feel free to 
modify it below:
"""
st.write(msg)
stop_words = copy(ENGLISH_STOP_WORDS)

is_stop = st.text_input("Is this word a stop word?").lower()
if is_stop != "":
    if is_stop in stop_words:
        st.write(f"Yes, `{is_stop}` is currently a stop word.")
    else:
        st.write(f"No, `{is_stop}` is not currently a stop word.")
    is_stop = ""

rm_word = st.text_input("Remove Stop Word").lower()
if rm_word != "":
    stop_words = stop_words.difference([rm_word])
    st.write(f"Removed `{rm_word}` from list.")
    rm_word = ""

add_word = st.text_input("Add Stop Word").lower()
if add_word != "":
    stop_words = stop_words.union([add_word])
    st.write(f"Added `{add_word}` to the list.")
    add_word = ""

st.subheader("What to Count: N-Grams")
msg = """
We can choose to either count each word individually, or we can count word pairings. 
This can be useful, say, if we think it's more important to count the phrase `clean energy` 
than counting `clean` and `energy` separately. Play with the options below to see how they change the 
different *tokens* that will be counted by our vectorizer.
"""
st.write(msg)

ngrams_sel = st.radio(label="We should count:", options=["unigrams", "bigrams"], index=0)

test_sentence = "Your cat is so fluffy I could die!"
st.write(f"Free Text: `{test_sentence}`")

if ngrams_sel == "unigrams":
    ngram_range = (1, 1)
    ngrams_demo_tokens = ngram_demo(test_sentence, 1)
else:
    ngram_range = (1, 2)
    ngrams_demo_tokens = ngram_demo(test_sentence, 2)

st.write(f"N-Gram Tokens: `{'`, `'.join(ngrams_demo_tokens)}`")  


st.subheader("How to Count: Vectorizer")
msg = """
Next, we count these n-gram tokens. There are two approaches here. The first  
is to simply count how many times each token appears in the response. The most common words will have 
the highest count. This is a good approach if we expect one topic to be dominant.

The second approach is to first count how many times a token appears in a response, and then divide 
that count by how many times the token appears in _all_ responses. This approach is called 
_Term Frequency-Inverse Document Frequency (TFIDF)_.

TFIDF "squishes" common words and highlights rare words. For example, if a response mentions the word 
_infrastructure_, how significant that is probably depends on how many other responses mention the word 
_infrastructure_ as well. If every other response mentions that word, it probably isn't notable. But if 
only a few others do, than perhaps it's something we should notice. This will be useful if we instead 
find a variety of topics among the responses, and want to measure the words that distinguish these topics. 
We don't yet know what we'll find, so we can try both.
"""
st.write(msg)

prepared_matrices = load_pickled_object("prepared_matrices.pkl")
vectorizer_choice = st.radio(label="Vectorizer to Use:", options=["Count Vectorizer", "TFIDF Vectorizer"])

vectorizer = prepared_matrices[question][vectorizer_choice][ngram_range]["vectorizer"]
X = prepared_matrices[question][vectorizer_choice][ngram_range]["X"]

sum_words = X.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

st.write("Top 20 Most Frequent Words among All Responses:")
st.write(f"`{'`, `'.join([t[0] for t in words_freq[:20]])}`")


st.header("Step 4: Latent Dirichlet Allocation")
msg = f"""
Latent Dirichlet Allocation (LDA) is a form of dimensionality reduction. Without getting too much into the details, 
dimensionality reduction is when we mathematically try to cut out the noise and reduce a dataset to just the 
values that provide meaning. It is a form of unsupervised machine learning, called "unsupervised" because 
we do not have the "correct" answers by which we can judge how well it did.

Here, we first specify how many topics we'd like the algorithm to search for. It then calculates that many 
groupings of words that more often appear together than appear with words from other groupings. 

Visually, imagine you were one of those psycopaths from the movies that had a room full of pictures and string 
tacked up on your walls. But now imagine that at one end of the room you had all the responses to the survey, 
at the other end you had every word that appears, and then you tacked up string from each response to each word it 
contained. You can image thick, tangled braids forming as lots of words point to the same responses. LDA is saying, 
"Q told me there's five topics!" and then grabbing the five thickests chords of string and posting your discovery on 
[Reddit](https://www.theatlantic.com/technology/archive/2020/09/reddit-qanon-ban-evasion-policy-moderation-facebook/616442/). 
We would need a lot of string though, because our {X.shape[0]} wonderful respondents wrote {X.shape[1]} different words!
"""
st.write(msg)

frequencies = X.sum(axis=0)
word_freq = [(word, idx, frequencies[:,idx][0,0]) for word, idx in vectorizer.vocabulary_.items()]
word_freq = sorted(word_freq, key=lambda w: w[2], reverse=True)

msg = """
We would need a lot of string
"""


n_components = st.slider(label="No. Topics", min_value=1, max_value=20, value=5)

lda = LatentDirichletAllocation(n_components=n_components, random_state=42)    
x = lda.fit_transform(X)
topic_keywords = retrieve_topic_keywords(lda, vectorizer.get_feature_names(), 10)

st.subheader("Topic Keywords")
for ii in topic_keywords:
    st.write(f"Topic {num2words(ii+1).title()}:\t`{'`, `'.join(topic_keywords[ii])}`")


st.header("Sandbox")
st.write("Here are all the options so you can play with making topics!")
st.subheader("Model Summary")

sandbox_question = st.radio(label="Free Response Question to Analyze", options=list(free_response_questions.values()), key="sandbox_question")
sandbox_question = reverse_col_to_key[sandbox_question]
sandbox_vect_choice = st.radio(label="Vectorizer to Use:", options=["Count Vectorizer", "TFIDF Vectorizer"], key="sandbox_vect_choice")
sandbox_ngrams_sel = st.radio(label="We should count:", options=["unigrams", "bigrams"], index=0, key="sandbox_ngrams_sel")
sandbox_components = st.slider(label="No. Topics", min_value=1, max_value=20, value=5, key="sanbox_components")

if sandbox_ngrams_sel == "unigrams":
    sandbox_ngram = (1, 1)
else:
    sandbox_ngram = (1, 2)


sand_vectorizer = prepared_matrices[sandbox_question][sandbox_vect_choice][sandbox_ngram]["vectorizer"]
X_sand = prepared_matrices[sandbox_question][sandbox_vect_choice][sandbox_ngram]["X"]
sandbox_model = LatentDirichletAllocation(n_components=sandbox_components, random_state=42)
x_sand = sandbox_model.fit_transform(X_sand)
sandbox_keywords = topic_keywords = retrieve_topic_keywords(sandbox_model, sand_vectorizer.get_feature_names(), 10)

for ii in sandbox_keywords:
    st.write(f"Topic {num2words(ii+1).title()}:\t`{'`, `'.join(sandbox_keywords[ii])}`")


st.header("Step 5: Assigning Responses to Topics")
msg = """


Now that we've generated topics, we'll be able to assign each response to whichever topic best describes it.
"""
st.write(msg)

st.subheader("Coming Soon!!")


if __name__ == "__main__":
    pass
    # import pickle

    # prepared_matrices = rec_dd()
    # print(prepared_matrices)

    # for question in free_response_questions:
    #     print(question)
    #     for vectorizer_choice in ["Count Vectorizer", "TFIDF Vectorizer"]:
    #         print(vectorizer_choice)
    #         for ngram_range in [(1, 1), (1, 2), (1, 3)]:
    #             print(ngram_range)

    #             if vectorizer_choice == "Count Vectorizer":
    #                 vectorizer = CountVectorizer(ngram_range=ngram_range)
    #             else:
    #                 vectorizer = TfidfVectorizer(ngram_range=ngram_range)

    #             X = vectorizer.fit_transform(analysis_dfs[question].processed)
    #             prepared_matrices[question][vectorizer_choice][ngram_range]["vectorizer"] = vectorizer
    #             prepared_matrices[question][vectorizer_choice][ngram_range]["X"] = X

    # pickle.dump(prepared_matrices, open("prepared_matrices.pkl", "wb"))
