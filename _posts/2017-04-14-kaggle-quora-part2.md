---
title: RNN for Quora duplicate questions
date:   2017-04-14 14:20:00 +0300
layout: post
comments: true
---

This is a follow-up post after [this one]({{ site.basurl }}{% post_url 2017-04-07-kaggle-quora-part1 %}) where I started participating in [Kaggle Quora competition](https://www.kaggle.com/c/quora-question-pairs). In this post I switched entirely to neural network based approaches to solve posed problem. Below are solutions I tried and submitted.

## Previous post summary

This week was started with a TF-IDF + SVD model that achieves 0.44 loss and have 0.79 accuracy. This week goal was to try deep learning solutions to tackle this problem without burden of manual feature engineering. Why bother if we can just throw some data (that is provided) into neural network and collect profits. Easy!

## Solutions found on the web

Before stepping to _"innovative"_ model architectures by myself, I've digged internet for already published solutions. Basically I found two solutions that tried to train-modify-and-submit.

 1. [https://github.com/bradleypallen/keras-quora-question-pairs](https://github.com/bradleypallen/keras-quora-question-pairs)
 1. [https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question](https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question)

### Solution #1

First solution is courtesy of `bradleypallen`. His repo contains very nice research on different approaches to tackle exactly the same problem. It is well structered and results are very accessible. Nice job!

Neural network used has a following architecture:

![braleypallen architecture](https://github.com/bradleypallen/keras-quora-question-pairs/raw/master/quora-q-pairs-model.png)

Really simple: represent question as a sequence of GloVe embeddings (300-dimensional), pass it to unidirectional LSTM layer, aggregate result, then pull it through several fully connected layers and sigmoid to get probabilty.

Author presents section called "Discussion" that is helpful to understand why he had chosen exactly this architecture. It's interesting to read it.

Training 1 epoch takes 1 minute with NVIDIA Titan X Pascal, training with default parameters (25 epochs) takes 25 minutes (obviously). Model overfits, and I'm going to stop at 10 epochs as results are already good enough: logloss is 0.3944, accuracy is 0.8255 (almost on par with what readme says). Submitting: kaggle loss is 0.379. That is improvement! Position improved by 80 to 645.

Author claimed that using dropout decrease accuracy. But I aimed for loss score rather than accuracy so I tweaked the model a little with dropout (between fully connected layers, rate is 0.2) and was able to slightly address overfitting issue with loss improvement (though accuracy suffered). Also, I've changed maximum length sequence to 40 rather than 25 to see if it would help.

With such settings model achieves loss 0.3840 and accuracy is 0.8263. It still overfits, though. Submission reveals small improvement with kaggle loss being 0.3765.

### Solution #2

Another solution I've encountered comes from `abhishekkrthakur` with his deep neural network that combines LSTM's and convolutions.

![abhishekkrthakur architecture](https://media.licdn.com/mpr/mpr/AAEAAQAAAAAAAAoQAAAAJDMyYzgyYzRhLTg1NjUtNDg1Yy1hMjY0LTJjZWFiMzJkOTk1NQ.png)

Model is bigger than our prevoius candidate, it has 3 different ways of question _"encoding"_ (think again as feature generation): 1 unidriectional LSTM encoder, 1 unidirectional LSTM encoder with aggregation (`TimeDistributed` and `Lamda` layers at the image) and convolutions path. Author claims it achieves 0.85 accuracy! That's impressive.

Firstly, I just tried to reproduce the results. Unfortunately, it takes pretty long to train: I needed 8 hours to train it for 200 epochs (default settings).

Implementation really achieves 0.848 accuracy on validation set, but it's susceptible to overfitting. Train accuracy is 0.99 and log loss in negligible. I tried to submit results to kaggle but result wasn't impressive: 0.6 loss. I should have tried to submit the solution after successful combat with overfitting but decided not to do so. At least, accuracy is _very_ good.

As a side note, [original post](https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur) contains very useful features that can be used in non neural networks classifiers or with combination. This blog post is insightful and offers useful knowledge to gain.

### Solution #3

My third approach was not inspired by any model or based on some other existing solution. Rather I've just decided to stop playing with other people solutions (and struggle with frameworks) and start from scratch. I had some experience with `chainer` and now it looked as good option for very quick prototyping.

I started with very simple RNN architectire: 1 layer of LSTM encoder and 2 fully connected layers on top of that. Very simple and easy to implement. Here is the architecture of it:

![Simple model architecture]({{site.url}}/assets/kaggle_simple_model.png)

Really simple: turn question into sequence of GloVe embeddings, pass them through LSTM embedding layer to get question representation as vector (I used 100-dimensional vectors), merge two vectors into one and pull it through 2 fully connected layers and softmax. No manual feature engineering and model architecture thinking.

```python
class SimpleModel(chainer.Chain):
    def __init__(self, vocab_size, in_dim, hidden_dim, dropout=0.0):
        super().__init__(
            q1_embedding=L.LSTM(in_dim, hidden_dim),
            q2_embedding=L.LSTM(in_dim, hidden_dim),
            fc1=L.Linear(2 * hidden_dim, hidden_dim),
            fc2=L.Linear(hidden_dim, 2),
        )
        self.embed = L.EmbedID(vocab_size, in_dim)
        self.dropout = dropout
        self.train = True

    def __call__(self, x1, x2):
        x1 = self.embed(x1)
        x2 = self.embed(x2)

        self.q1_embedding.reset_state()
        self.q2_embedding.reset_state()
        seq_length = x1.shape[1]
        q1 = q2 = None
        for step in range(seq_length):
            q1 = self.q1_embedding(x1[:, step, :])
            q2 = self.q2_embedding(x2[:, step, :])

        x = F.concat([q1, q2], axis=1)
        x = F.relu(self.fc1(F.dropout(x, self.dropout, self.train)))
        x = F.relu(self.fc2(F.dropout(x, self.dropout, self.train)))
        x = self.fc4(F.dropout(x, self.dropout, self.train))

        return x
```

I set `in_dim` to 300 (GloVe embedding dimensionality), `hidden_dim` to 100, `dropout` to 0.2. Model was trained with Adam (learning rate is 0.001) for 15 epochs with a batch size 128. Accuracy was 0.83 and loss is 0.446. Submission on kaggle earned loss of 0.4543. Obviously, this is not an improvement over solution #1 (and architecture is very similiar) but it's simple enough, trains fast (again, less than minute per epoch) and provides a good baseline for further experiments.

Due to model small size I was able to play with batch sizes (Titan X has 12 GB of memory). I could even train the network with a 8192 batch size! That's big number. During experiments I've found that smaller batch sizes allow actually to get better accuracies.

I then tried to train the network for 100 epochs. It overfitted, but i managed to obtain good enough score at 6 epoch with a loss of 0.375196 and accuracy 0.832. Early stopping in action :) Submission was pretty succesful: kaggle loss was 0.36134 that is improvement and position jumped up again, to 659/1474! Not bad for 5 minutes of training. During my experiments ~350 new participants entered the competition, so I intently specifying position with two numbers: for better progress tracking.

However, I felt that more complex model should perform even better. The problem is, I _don't_ know what architecture this model should have ;)

## Solution #4

I then switched to multilayer bidirectional LSTM encoder (instead of 1 layer unidirectional). Instead of representing question as one vector with 1 LSTM cell we now have two vectors (bi-) from 4 LSTM cells. Here what question (not word) embedding looked like before:

![1 layer unidirectional LSTM]({{site.url}}/assets/kaggle_lstm_layer.png)

And this is what happens with 4 layers:

![Multilayer biidirectional LSTM]({{site.url}}/assets/kaggle_multilayer_bilstm.png)

Model definition (and implementation) in `chainer` looks like this:

```python
class SimpleModel(chainer.Chain):
    def __init__(self, layer_num, vocab_size, in_dim, hidden_dim, dropout=0.0):
        super().__init__(
            f_embedding=L.NStepLSTM(layer_num, in_dim, hidden_dim, dropout),
            b_embedding=L.NStepLSTM(layer_num, in_dim, hidden_dim, dropout),
            fc1=L.Linear(4 * hidden_dim, hidden_dim),
            fc2=L.Linear(hidden_dim, hidden_dim),
            fc3=L.Linear(hidden_dim, hidden_dim),
            fc4=L.Linear(hidden_dim, 2),
        )
        self.embed = L.EmbedID(vocab_size, in_dim)
        self.dropout = dropout
        self.train = True

    def __call__(self, x1, x2):
        sections = np.cumsum(np.array([len(x) for x in x1[:-1]], dtype=np.int32))
        x1 = F.split_axis(self.embed(F.concat(x1, axis=0)), sections, axis=0)

        _, _, q1_f = self.f_embedding(None, None, x1, self.train)
        _, _, q1_b = self.b_embedding(None, None, x1[::-1], self.train)

        q1_f = F.concat([x[-1, None] for x in q1_f], axis=0)
        q1_b = F.concat([x[-1, None] for x in q1_b], axis=0)

        sections = np.cumsum(np.array([len(x) for x in x2[:-1]], dtype=np.int32))
        x2 = F.split_axis(self.embed(F.concat(x2, axis=0)), sections, axis=0)

        _, _, q2_f = self.f_embedding(None, None, x2, self.train)
        _, _, q2_b = self.b_embedding(None, None, x2[::-1], self.train)

        q2_f = F.concat([x[-1, None] for x in q2_f], axis=0)
        q2_b = F.concat([x[-1, None] for x in q2_b], axis=0)

        x = F.concat([q1_f, q2_f, q1_b, q2_b], axis=1)
        x = F.relu(self.fc1(F.dropout(x, self.dropout, self.train)))
        x = F.relu(self.fc2(F.dropout(x, self.dropout, self.train)))
        x = F.relu(self.fc3(F.dropout(x, self.dropout, self.train)))
        x = self.fc4(F.dropout(x, self.dropout, self.train))

        return x
```

I'm not very proud of such implementation (I have to struggle to make `NStepLSTM` work), speed dropped drastically and I wasn't able to utilize GPU fully. Implementation still needs lots of thought and effort to be useable. Training for 10 epochs takes 1.5 hours that is much slower than previous approach. I guess I'm not doing something quite right here. The only advantage over previous model is that now one don't have to crop and pad input sequences (because of `NStepLSTM` implementation) but can pass them as is, with full length. Also, it uses cuDNN optimized multilayer rnn implementation that should also be beneficial. Unfortunately, this is only one part of a story.

Anyway, with 4 `layer_num`, 100 `hidden_dim`, 0.2 `dropout` model was trained for 16 epochs (trained for 50 actually, it started overfitting after 16) validation loss was at 0.4125, kaggle loss after submission was 0.4149 that wasn't an improvement.

## Other tricks

I want to share some tricks people on kaggle actively use:

 * Class weights
 * Ensembling

There are ongoing discussions on kaggle forums that both train and test sets unbalanced: positive samples ratio is far from `0.5`. Moreover, it's said that train and test sets have different positive-to-negative ratios. That might hurt classifier performance and our sumbission score. For example, read [this thread](https://www.kaggle.com/c/quora-question-pairs/discussion/31179) for more information.

I used this trick on one of my model (4-layer LSTM) and it improved my score from 0.414 to 0.380. Significant gain for probabilities renormalizing.

Another technique that can be used to boost one's score is to ensemble your models. There are mentions from many people that ensemble helps to boost score another 0.02-0.03 points. Read more about that [in this thread](https://www.kaggle.com/c/quora-question-pairs/discussion/31186).

Also, I wish I have time (and patience) to perform proper hyper-parameter tuning. Rumors are it can also boost performance by another 0.0X points!

## Conclusions

Neural networks can really help with such tasks as discussed solutions demonstrate. However, there is no silver bullet and much more efforts need to be invested to obtain top-level results. Consider, for instance, [this repo](https://github.com/ChenglongChen/Kaggle_HomeDepot) with a very similiar competition.
