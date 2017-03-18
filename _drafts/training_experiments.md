---
layout: post
title: Training neural voice models
date: 2017-03-04 14:00 +0300
---

On the wave of interesting voice related papers one could be interested what results could be delivered of current deep neural networks for various voice tasks: namely, speech recognition (ASR), and speech (or just audio) sythesis. Ultimately, text-to-speech (TTS) models (famous Deepvoice, for example) is also very intriguing but since there is no open source implementation (yet) no results for it.

## Introduction

In this particular blog post I wan't to share my experience training different models. All models are publicly available at github.

 1. https://github.com/buriburisuri/speech-to-text-wavenet
 1. https://github.com/SeanNaren/deepspeech.pytorch
 1. https://github.com/soroushmehr/sampleRNN_ICLR2017
 1. (maybe parrot - char2wav?)

## ASR

Automatic speech recognition (ASR) task is to convert raw audio sample into text. That is, simple speech-to-text conversion: given raw audio file as input model should output text (ascii symbols) of corresponding text.

#### Wavenet

For that purpose I used `buriburisuri` implementation of wavenet paper for speech recongiton. Implementation details could be easily found in a repo. The model was trained with VCTK dataset that provides both raw audio and text transcripts. Training took 38 hours on 1 NVIDIA Titan X Pascal GPU for 45k iterations with default parameters.

![Loss function]({{site.url}}/assets/2017-03-18-175047_323x255_scrot.png)

Training instructions are available in repo README so I won't repeat them here.

Here are results:

(results from speech-to-text-wavenet-it's on dnn3 machine, busy right now with deepspeech)

#### Deepspeech2

Other model that also performs speech-to-text conversion comes from `SeanNaren`. He has baidu deepspeech2 model implementaion in pytorch. Getting this to work required some efforts because of using `warp-ctc` bindings from baidu. These are highly GPU and CPU optimized operations for calculating CTC loss that is used in both models.

To get that to work I had to also compile pytorch from sources as it was built against `libgomp` of other version and `import warp_ctc` command failed with

    path-to-pytorch-included-libs/libgomp.so.1: version `GOMP_4.0' not found (required by path-to-built-warpctc/libwarpctc.so.0)

Obviously, `libwarpctc` was linked against `libgomp` of a different version than a pytorch binary libraries (there are bunch of libs in pytorch distributions). To fix that I had to built pytorch from sources as well (to link it against the same lib). So the whole path I need to take was:

 1. Install `torch` (from sources, required by `warp-ctc` library)
 1. Install `pytorch` (from sources)
 1. Compile and install `warp-ctc`
 1. Profit, finally.

Original implementation uses `an4` dataset for model training. Dataset is not big and trainging takes about an hour with default settings. Here are results for `an4` dataset:

(results)

However, I was also interested in training the model with VCTK (as a little larger dataset). Implementation provides simple instruction how to convert dataset to train: just provide manifest file (that is, simple csv with path to `wav` file and path to corresponding transcription). It took me a while to preprocess dataset to comply to the model. Overall process is as follows:

 1. Download VCTK
 1. Downsample it to 16k (or change default settings and model definition a little)
 1. Create manifest file
 1. Modify source to properly parse transcription text files

Following snippet uses `zsh` globbing feature:

```sh
mkdir dataset/; cd dataset/
wget http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz

for file in dataset/wav48/**/*.wav; do ffmpeg -i $file -ac 1 -ar 16000 -acodec pcm_s16le "${file/wav48/wav16}" ; done
```

Create manifests with ipython:

```python
import os
import random

path = 'dataset/'

wav_files = [os.path.join(root, filename) for root, _, files in os.walk(os.path.join(path, 'wav16')) for filename in files]
txt_files = [file.replace('wav16', 'txt').replace('.wav', '.txt') for file in wav_files]
data = ['{},{}\n'.format(fst, snd) for fst, snd in zip(wav_files, txt_files)]

random.shuffle(data)
S = int(len(data) * 0.9)
train, test = data[:S], data[S:]

with open('train.csv', 'w') as f:
    f.writelines(train)
with open('test.csv', 'w') as f:
    f.writelines(test)
```

Unfortunately, there is no transciption for `p315` entries. We remove those from train and test sets.

```sh
sed /p315/d train.csv > train_m.csv
sed /p315/d test.csv > test_m.csv
```

To make data loading work one has to modify source code. I also use `python3` so modifications reflect that as well:

```diff
diff --git a/data/data_loader.py b/data/data_loader.py
index dcea23e..b24ff9c 100644
--- a/data/data_loader.py
+++ b/data/data_loader.py
@@ -1,3 +1,4 @@
+import string
 import librosa
 import numpy as np
 import scipy.signal
@@ -92,9 +93,14 @@ class SpectrogramDataset(Dataset, SpectrogramParser):
         return spect, transcript

     def parse_transcript(self, transcript_path):
+        translator = str.maketrans('', '', string.punctuation)
         with open(transcript_path, 'r') as transcript_file:
-            transcript = transcript_file.read().replace('\n', '')
-        transcript = [self.labels_map[x] for x in list(transcript)]
+            transcript = transcript_file.read().replace('\n', '').replace('\t', '').translate(translator)
+        transcript = list(transcript)
+        try:
+            transcript = [self.labels_map[x.upper()] for x in transcript]
+        except KeyError as e:
+            print(e, transcript)
         return transcript

     def __len__(self):
diff --git a/data/utils.py b/data/utils.py
index 8ea1bd0..0866dbc 100644
--- a/data/utils.py
+++ b/data/utils.py
@@ -32,7 +32,7 @@ def create_manifest(data_path, tag, ordered=True):
         for wav_path in file_paths:
             transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
             sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
-            file.write(sample)
+            file.write(bytes(sample, 'utf-8'))
             counter += 1
             _update_progress(counter / float(size))
     print('\n')
diff --git a/decoder.py b/decoder.py
index 155a5e4..4f6038b 100644
--- a/decoder.py
+++ b/decoder.py
@@ -40,7 +40,7 @@ class Decoder(object):
     def convert_to_strings(self, sequences, sizes=None):
         """Given a list of numeric sequences, returns the corresponding strings"""
         strings = []
-        for x in xrange(len(sequences)):
+        for x in range(len(sequences)):
             string = self.convert_to_string(sequences[x])
             string = string[0:int(sizes.data[x])] if sizes else string
             strings.append(string)
@@ -90,7 +90,7 @@ class Decoder(object):

         # build mapping of words to integers
         b = set(s1.split() + s2.split())
-        word2char = dict(zip(b, range(len(b))))
+        word2char = dict(list(zip(b, list(range(len(b))))))

         # map the words to a char array (Levenshtein packages only accepts
         # strings)
```

Training with default parameters fails for me because of exploding gradients. I have to reduce learning rate to start it going. I didn't perform any search, just decreased it a little.

    python train.py --train_manifest train_m.csv --val_manifest test_m.csv --epochs 200 --lr 1e-5


(How much it takes to train the model)
(results)

## Audio synthesis

For synthesis I took SampleRNN implementation from paper authors. Model traininig time is 1 week per experiment (as they state in a paper). First experiment used classic piano music (Bach dataset). Here are results.

(after 1 day of trainingg)
(after 3 day of trainingg)
(after 1 week of trainingg)
(piano)

For me it sounds really nice!

Next experiment is to use VCTK speech and let the algorithm do the magic. Note that model is unconditional so it won't learn how to properly spell the words. Rather we can expect it to be some _'alien'_ language slightly resembling english.

(results)
