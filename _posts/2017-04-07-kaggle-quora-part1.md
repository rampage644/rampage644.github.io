---
layout: post
title: Quora questions Kaggle competition
date: 2017-04-07 13:40 +0300
comments: true
---

I recently found that quora released first publicly available dataset: [question pairs][dataset]. Moreover, they also started [Kaggle competition](https://www.kaggle.com/c/quora-question-pairs) based on that dataset. In these blog posts series, I'll describe my experience getting hands-on experience participating in it.

## Introduction

The dataset basically consists of question pairs and your task is to detect duplicate pairs. That's simple. Here are few samples of records:

```
In [9]: for i in range(5):
   ...:     print(f'{data.question1[i]} - {data.question2[i]}')
   ...:
   ...:
What is the step by step guide to invest in share market in india? - What is the step by step guide to invest in share market?
What is the story of Kohinoor (Koh-i-Noor) Diamond? - What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?
How can I increase the speed of my internet connection while using a VPN? - How can Internet speed be increased by hacking through DNS?
Why am I mentally very lonely? How can I solve it? - Find the remainder when [math]23^{24}[/math] is divided by 24,23?
Which one dissolve in water quikly sugar, salt, methane and carbon di oxide? - Which fish would survive in salt water?

```

I was particularly interested in how good RNNs are compared to some other methods. For that, I decided to start with most simple and straightforward approaches.

Before stepping up to some coding let's first dive into some NLP theory. First, such tasks are properly called [_"Paraphrase Identification"_](https://aclweb.org/aclwiki/index.php?title=Paraphrase_Identification_(State_of_the_art)). Following the link will help you understand what current state of the art approaches are.

Let's try to figure out what kind of problems we can solve with such techniques. At a glance, it can be used in search and document ranking: one can compare how similar search query to a document or group of documents. Then we can compare what is the most similar document and rank those according to similarity. Another idea is grouping (or clustering) similiar documents into different categories: think of _"sports"_, _"economics"_, _"engineering"_. One can go even further and perform plagiarism detection. The even wilder idea is to use it on machine translation to compare phrases in different languages, that is, translate them.

## Coding

In order for machine learning algorithm to understand the text, we need somehow to convert it to numbers. Numbers are the only substance computers understand.

Basically, we want to turn those nasty strings into some integers and floats. How can we do that? Let's put aside mathematical-statistical-whatever techniques like [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) and [word embeddings](https://en.wikipedia.org/wiki/Word2vec) and try something simple.

Initially, I went through some kaggle kernels and topic threads to get a very high-level understanding how people solve problems like this. I found some very helpful and insightful kernels where one can find a handful of ideas (features) how to turn questions (strings) into numbers.

For first attempt I went with each question length, words count in each and a total number of common words in two questions: five features in total. Here is code snippet that does exactly that.

```python
import pandas as pd

def common_words(x):
    q1, q2 = x
    return len(set(str(q1).lower().split()) & set(str(q2).lower().split()))

def words_count(question):
    return len(str(question).split())

def length(question):
    return len(str(question))

data = pd.read_csv('data/train.csv')

data['q1_words_num'] = data['question1'].map(words_count)
data['q2_words_num'] = data['question2'].map(words_count)

data['q1_length'] = data['question1'].map(length)
data['q2_length'] = data['question2'].map(length)

data['common_words'] = data[['question1', 'question2']].apply(common_words, axis=1)
```

We load the data into `pandas` dataframe add create 5 new features out of the raw text.


Then I used random forest classifier (without any hyperparameter tuning, default parameters are good enough) to fit the model and get accuracy and score:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

train, val = train_test_split(data, train_size=0.8)

classifier = RandomForestClassifier(n_jobs=-1)
classifier.fit(X, y)

model.fit(train[['question1', 'question2']], train.is_duplicate)
score = model.score(val[['question1', 'question2']], val.is_duplicate)
print(f'Score is {score:.2f}')
```

For that minimal effort (well, not minimal - lots of reading and research before) score is 0.68. Score roughly translates to accuracy (later I calculated accuracy as well). Looks really good for me. However, before starting coding I studied Kaggle kernels and read a lot. So this feature selection is really good, and credit for that features goes to [Philipp Schmidt](https://www.kaggle.com/philschmidt).

My first submission! Wow, I'm not the last: current standing is 1080/1181. Log-loss is 9.92. Though, I scored pretty low at a leaderboard that gives me room for improvement. Nice environment for a _"flow"_.

My next idea to try was to do some data cleaning and preprocessing since its most "vital" part of any machine learning related task. Let's do some then!

What kind of preprocessing can we do for plain English? Some of the popular choices are:

 1. Punctuation removal
 1. Tokenization
 1. Stemming
 1. Lemmatisation
 1. Spelling correction

Punctiation removal is the most straightforward: one can do it without any ML or NLP knowledge:

```python
def remove_punctiation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
```

Other steps are more difficult. To perform _"proper"_ tokenization and stemming we will use nice NLP libraries: `spacy` and `nltk`. Well, let's use them to everyone's good. First, `nltk`-based solution:

```python
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# postpone your question why lemmatize :)
l = WordNetLemmatizer()
    def clean_nltk(x):
        x = str(x).lower()
        return ' '.join([l.lemmatize(t) for t in word_tokenize(x) if t not in stops and t not in string.punctuation])
```

I also tried to use `spacy` but found such option to be actually slower than previous one. This is because I wasn't doing it correctly (use `nlp.pipe()`) and whole operation does much more than just tokenization (well, it allows us to access `lemma_` of a token, there much more info):

```python
import spacy

nlp = spacy.load('en')
def clean(x):
    x = str(x)
    # losing info about personal pronouns here (due to lemmatization)
    return ' '.join([token.lemma_ for token in nlp(x) if not token.is_punct and not token.is_stop])
```

Later, I replaced lemmatisation with stemming but it didn't change anything with the current setup. Moreover, `nltk` have some [bug](https://github.com/nltk/nltk/issues/1581) that didn't allow me to use its stemmer so I had to switch to `PyStemmer`. My final preprocessing step looked like this:

```python
from nltk import word_tokenize
from Stemmer import Stemmer

s = Stemmer('english')
translator = str.maketrans('', '', string.punctuation)
def tokenize(text):
    return s.stemWords(word_tokenize(text.translate(tr)))
```

However, doing some cleaning and preprocessing didn't help me to improve model score and my standing. I should find other ways to improve.

### Competition ranking criteria

As a side note, I want to bring one's attention to the fact how this particular competition ranks submissions. I noticed that my loss is very high, though accuracy is on par with numbers other guys report. How so?

The answer is that it uses log loss to evaluate submissions. It follows that one better to output probabilities rather than classes because loss penalizes heavily big errors. I changed my submission code a little and improved my kaggle loss from 9.92 to 1.0! Though my standing didn't improve much, only by 100 positions. Well, this was newbie error.

```diff
- X['is_duplicate'] = model.predict(X)[:, 1]
+ X['is_duplicate'] = model.predict_proba(X)[:, 1]
```

## TF-IDF plus LDA

Remember, our idea is to convert strings into numbers? Though current approach gives some results I think (as do other people) we can do much better. Let's explore how else can we _"featurize"_ texts.

I won't go into detail since everything is already written: [Bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model), [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), [blog post about tf-idf #1](http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/), [blog post about tf-idf #2](http://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/). In short, you convert words into numbers (integers or floats) based on documents you have (set of documents is called corpus). Each document (question in our case) is represented with a real-valued vector.

```python
import pandas as pd
from Stemmer import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer

tr = str.maketrans('', '', string.punctuation)
s = Stemmer('english')
def tokenize(text):
    return s.stemWords(nltk.word_tokenize(text.translate(tr)))

vectorizer = TfidfVectorizer(min_df=1, stop_words='english', tokenizer=tokenize)

data = pd.read_csv('data/train.csv')
data = vectorizer.fit_transform(data)
```

Essentially, this `vectorizer` just 'vectorizes' text: turns it into numbers. But those numbers are not hand-crafted by somebody but calculated. The bigger the corpus (set of documents) - the better.

What we have after this operation is a big _"co-occurrence"_ matrix. Its shape is `[number of documents X vocabulary size]`. The difference with our previous approach is that number of features increased: from 5 it went to `vocabulary size` (~86k after stopwords removal).

We can train a classifier on top of that, but I even didn't try it. Instead, I decided to go another way: [LSA](https://en.wikipedia.org/wiki/Latent_semantic_analysis) (latent semantic analysis).

The high-level idea is very simple: we assume that each document (question) has some latent (hidden) variables 'behind' that define its content and meaning, and we want to model those variables instead. We also hope that number of those latent variables would be smaller than vocabulary size.

Implementation! There is nice [tutorial](https://radimrehurek.com/gensim/tut2.html) on model we're about to implement (from `gensim` library). However, I will continue to use `sklearn`.

```python
import itertools
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('data/train.csv')

vectorizer = TfidfVectorizer(min_df=1, stop_words='english', tokenizer=tokenize)
svd_model = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=5)
svd_transformer = Pipeline([('tfidf', vectorizer),
                            ('svd', svd_model)])

g = itertools.chain(data.question1, data.question2)
svd_transformer.fit(g)

q1, q2 = transformer.transform(data.question1), transformer.transform(data.question2)
# Memory inefficient
data['cosine_distance'] = np.diag(cosine_similarity(q1, q2))
```

Here I use `Pipeline` for convenience: data is passed through vectorizer first and then goes into SVD (thus reducing dimensionality from whatever it was to `n_components=100`). Each question is now is a vector of 100 numbers and all we need to do is to combine two questions. I use cosine similarity as a metric. Right now our dataset contains only 1 (!) feature.

With only that feature my random forest classifier scores at 0.65. Apparently, it's not better than the previous model, so there is need for further efforts. What about adding features? :)

```python
data['l2_distance'] = np.linalg.norm(q1 - q2, axis=1)
data['l1_distance'] = np.abs(q1 - q2).sum(axis=1)
```

Two more features added: distance between question vectors. 3 features in total. Model scores at 0.72 that is even better than the previous model. Submission: kaggle loss is 0.88, position is 1081. We're moving!

But I'm too lazy engineering features again. What about just combining question vectors and let classifier do its job?

```python
data = pd.read_csv('data/train.csv')

vectorizer = TfidfVectorizer(min_df=1, stop_words='english', tokenizer=tokenize)
svd_model = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=5)
svd_transformer = Pipeline([('tfidf', vectorizer),
                            ('svd', svd_model)])

g = itertools.chain(data.question1, data.question2)
svd_transformer.fit(g)

q1, q2 = transformer.transform(data.question1), transformer.transform(data.question2)
data = pd.DataFrame(q1 - q2)
```

That will give us 100 features. Model scores at 0.75 (improvement again), kaggle loss is 0.448 and position is 852/1270. I decided to add more features by different ways of combining vectors: L1 distance, L2 distance and elementwise multiplication (sort of angle):

```python
data = pd.DataFrame(np.concatenate([np.abs(q1 - q2), np.sqrt((q1 - q2) ** 2),  q1 * q2], axis=1))
```

That model scores at 0.79, kaggle loss is 0.44, the position goes up to 765/1281.

## Conclusion

Meantime I tried other options: using 2-3-4-gram instead of just unigrams, parameters searching through cross-validation, using different classifiers. None of them gave me improvement. According to a leaderboard, people are doing much better than me, so I'll continue looking for a better way to tackle the problem. In the next blog post, I'll switch entirely to neural networks and continue working with them.


[dataset]: https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs
