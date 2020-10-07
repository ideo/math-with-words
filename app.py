import pandas as pd
import streamlit as st
# from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from num2words import num2words

from aws import load_pickled_dataframe, load_pickled_object
# from nlp_pipeline import NLP, load_spacy, nlp_pipeline, ngram_demo, retrieve_topic_keywords
from nlp_pipeline import ngram_demo, retrieve_topic_keywords


st.title("Hello, Koko!")
opening_msg = """
I made this so we could explore the survey results together! Your question for me was, 
"what's the typical response?". Here's how we'll go about answering that. First, we'll 
do some topic modeling, to see what are the major themes people are talking about. 
Then, we'll assign each response to the topic or topics that best describes it. From 
there, we can see how many people are talking about what.

The tricky thing about topic modeling is that there is no mathematical way to know if 
you've found the "correct" number of topics. We simply need to read through the results 
and decide for ourselves if they make sense. So hopefully this walkthrough serves as a 
way for us to do that together! Welcome to Topic Modeling 101!
"""
st.write(opening_msg)

# Translation
st.header("Step 1: Translation")
msg = """
How's it looking? Nothing I can really do here so hope it's good! 
We'll keep the Japanese along throughout so you can do the sanity check for us. All hail DeepL.
"""
st.write(msg)

# filepath = "../data/attitudes_survey/attitudes_survey_translation_9_25.csv"
# df = pd.read_csv(csv_file)
# df.drop(columns=["Unnamed: 0"], inplace=True)
df = load_pickled_dataframe()
st.dataframe(df)


st.header("Step 2: Tokenization")
msg = """
Topic Modeling is basically counting words. So the first step here is to modify 
the words a bit so they're easier to count.
"""
st.write(msg)

st.subheader("Stop Words")
msg = """
First, we're going to remove very common, meaningless words such as `the` and `in` called *stop words*. Because if not, 
they will be counted most often and our topics will be dominated by meaningless words.

The library I'm using includes a helpful stop words list. But I'm going to modify it a bit. It includes the word 
`out` and I noticed that after removing stop words a response, "I am out of money", was turned into, "I am money".
Slightly different meaning. I've also chosen to remove negative words, like `no`, `not` and `none` from the list. 
So the list is not exact. Feel free to modify it below.
"""
st.write(msg)

# is_stop = st.text_input("Is this word a stop word?").lower()
# if is_stop != "":
#     if NLP.vocab[is_stop].is_stop:
#         st.write(f"Yes, `{is_stop}` is currently a stop word.")
#     elif ~NLP.vocab[is_stop].is_stop:
#         st.write(f"No, `{is_stop}` is not currently a stop word.")
#     is_stop = ""

# rm_word = st.text_input("Remove Stop Word").lower()
# if rm_word != "":
#     NLP.vocab[rm_word].is_stop = False
#     st.write(f"Removed {rm_word} from list.")
#     rm_word = ""

# add_word = st.text_input("Add Stop Word").lower()
# if add_word != "":
#     NLP.vocab[add_word].is_stop = True
#     st.write(f"Added {add_word} from list.")
#     add_word = ""


st.subheader("Lemmatization")
msg = """
Next, computers are dumb. The words `run`, `running`, and `ran` will be counted as differnt words.
So the big thing we're going to do here is a process called lemmatization, where we reduce all 
words to their _lemma_, such as `run`.

These steps together, along with breaking up contractions, removing punctuation, and converting everything, are 
called "Tokenization". We reduce the text to a list a valid "tokens" we think are useful to analyze.
"""
st.write(msg)


temp_msg = """
I wanted to let you play with this part, but the [model](https://spacy.io/) that does the tokenization 
is like 50 gigs and way to big to host on heroku for no money. There's more AWS stuff I'll need to 
learn to get that working. So for now, here are the results of the process I ran.
"""
st.write(temp_msg)


free_response_questions = {
    "holding_back": "If you want to contribute more to the environment, I'd like to ask you -- what do you feel is currently holding you back from taking action to contribute more?",
    "habits": "Are there any actions or habits that you have devised that are unusual for those around you that lead to eco/environmental issues?",
    }
chosen_col = st.radio(label="Free Response Question to Analyze", options=list(free_response_questions.values()))

reverse_col_to_key = {v:k for k,v in free_response_questions.items()}
analysis_dfs = load_pickled_object("analysis_dfs.pkl")
analysis_df = analysis_dfs[reverse_col_to_key[chosen_col]]

# analysis_df = df[[chosen_col]].rename(columns={chosen_col: "raw"})

# clicked = st.button("Process Text")
# if clicked:
#     analysis_df = nlp_pipeline(analysis_df, NLP)
#     st.dataframe(analysis_df[["raw", "processed"]], width=700)
st.dataframe(analysis_df[["raw", "processed"]], width=700)


st.header("Step 3: Vectorization")
msg = """
Computers do math with numbers and not words. Here is where we count the words. Two important concepts here: 
what we're counting and how we're counting them.
"""
st.write(msg)

st.subheader("What to Count: N-Grams")
msg = """
We can choose to either count each word individually, or we can count word pairings or word triplets. 
This can be useful, say, if we think it's more important to count the phrase `clean energy` 
that `clean` and `energy` separately. Play with the options below to see how they change the 
different tokens that will be counted by our vectorizer.
"""
st.write(msg)

ngrams_sel = st.radio(label="We should count:", options=["unigrams", "bigrams", "trigrams"], index=1)

test_sentence = "Your cat is very fluffy."
st.write(f"Free Text: `{test_sentence}`")

if ngrams_sel == "unigrams":
    ngram_range = (1, 1)
    ngrams_demo_tokens = ngram_demo(test_sentence, 1)
elif ngrams_sel == "bigrams":
    ngram_range = (1, 2)
    ngrams_demo_tokens = ngram_demo(test_sentence, 2)
else:
    ngram_range = (1, 3)
    ngrams_demo_tokens = ngram_demo(test_sentence, 3)

st.write(f"N-Gram Tokens: `{'`, `'.join(ngrams_demo_tokens)}`")  


st.subheader("How to Count: Vectorizer")
msg = """
Next, we count each word to form a vector. There are two approaches here. The first approach 
is to simply count how many times each word appears in the response. The second approach is to first 
count how many times a word appears in a response, and then divide that count by how many times the word 
appears in _all_ responses. This approach is called _Term Frequency-Inverse Document Frequency (TFIDF)_.

TFIDF "squishes" common words and highlights rare words. For example, if a response mentions the word 
_infrastructure_, how significant that is probably depends on how many other responses mention the word 
_infrastructure_ as well. If every other response mentions that word, it probably isn't notable. But if 
only a few others do, than perhaps it's something we should notice.

TFIDF is useful when you're expecting to see a variety of topics in the collection. Simply counting 
is useful when you're expecting everyone to be writing about the same topic. I'm not sure which to expect 
here, so we can try both.

The vector we create will be as long as the number of unique words that appear in all of the survey responses. 
For each response, we'll fill in each word's count in the spot in the vector corresponding to that word. Since 
the responses aren't terribly long, and there's a lot of words overall, the vectors will be mostly filled with zeros.
"""
st.write(msg)

vectorizer_choice = st.radio(label="Vectorizer to Use:", options=["Count Vectorizer", "TFIDF Vectorizer"])

if vectorizer_choice == "Count Vectorizer":
    vectorizer = CountVectorizer(ngram_range=ngram_range)
else:
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)


# If we skip to the bottom, this section won't have run
if "processed" not in analysis_df.columns:
    analysis_df = nlp_pipeline(analysis_df, NLP)

X = vectorizer.fit_transform(analysis_df.processed)


st.header("Step 4: Dimensionality Reduction")
msg = """
Now we do some linear algebra!
"""
st.write(msg)

n_components = st.slider(label="No. Topics", min_value=1, max_value=20, value=5)
lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
x = lda.fit_transform(X)
topic_keywords = retrieve_topic_keywords(lda, vectorizer.get_feature_names(), 10)

st.subheader("Model Summary")
msg = f"""
* Survey Question:\t**{chosen_col}**
* N-Gram Range:\t**{ngrams_sel.title()}**
* Vectorizier:\t**{vectorizer_choice}**
* No. Topics:\t**{num2words(n_components).title()}**
"""
st.write(msg)

st.subheader("Topic Keywords")
for ii in topic_keywords:
    st.write(f"Topic {num2words(ii+1).title()}:\t`{'`, `'.join(topic_keywords[ii])}`")


st.header("Step 5: Assigning Responses to Topics")
msg = """
Now that we've generated topics, we'll be able to assign each response to whichever topic best describes it.
"""
st.write(msg)

st.subheader("Coming Soon!!")
