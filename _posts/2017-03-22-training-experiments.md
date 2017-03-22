---
layout: post
title: Training neural models for speech recognition and synthesis
date: 2017-03-22 16:30 +0300
---

On the wave of interesting voice related papers, one could be interested what results could be achieved with current deep neural network models for various voice tasks: namely, speech recognition (ASR), and speech (or just audio) synthesis. Ultimately, [Deepvoice][deepvoice] text-to-speech (TTS) model is very intriguing but since there is no publicly available implementation (yet) one can't perform experiments with it.

## Introduction

In this particular blog post, I'd like to share my experience training different models that are related to various voice specific tasks. All models used are publicly available at Github: either it's the code accompanying the paper or just published implementation by some smart and generous people (third party implementations).

I was interested what results could already be achieved using open datasets and without access to clusters of 8 Titan X packed nodes. Here are implementations used in this experiment:

 1. [Wavenet ASR][wavenet-asr]
 1. [Deepspeech2 ASR][deepspeech2-asr]
 1. [SampleRNN](https://github.com/soroushmehr/sampleRNN_ICLR2017)

## ASR

Automatic speech recognition (ASR) task is to convert raw audio sample into text. That is, simple speech-to-text conversion: given raw audio file as input, model should output text (ASCII symbols) of corresponding text.

#### Wavenet

For that purpose, I used `buriburisuri` [implementation][wavenet-asr] of wavenet paper for speech recognition. Implementation details could be easily found in a repo. The model was trained with VCTK dataset that provides both raw audio and text transcripts. Training took 38 hours on 1 NVIDIA Titan X Pascal GPU for 45k iterations with default parameters.

![Loss function]({{site.url}}/assets/2017-03-18-175047_323x255_scrot.png)

Training instructions are available in repo [README](https://github.com/buriburisuri/speech-to-text-wavenet/blob/cfacb1bc8a8d33478b201ff34cd3847b3c8cd891/README.md) so I won't repeat them here.

Here are results:

    # ground truth is 'Please call Steve'
    $ python recognize.py --file asset/data/wav48/p225/p225_001.wav
    these p call stever

    # ground truth is 'She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.'
    $ python recognize.py --file asset/data/wav48/p236/p236_005.wav
    she can scop these things into three red bags and e will ge mee her ledngesdayathetrain station

    # ground truth is 'I wondered about changing back to rugby.'
    $ python recognize.py --file asset/data/wav48/p317/p317_063.wav
    i wondered aboute chaning bac to rugbey

Not bad at all. Results are not perfect, for sure, and modern ASR systems are much better but we've trained the model on _really_ small dataset for only 30 hours.

**Update**: repo code was updated to work with other datasets, data augmentation. It's possible that training it right now would produce much better results. However, I haven't repeated it. In case one is interested there is a link to trained model in a README.

#### Deepspeech2

Another model that also performs speech-to-text conversion comes from `SeanNaren`. He has [implemented][deepspeech2-asr] Baidu [deepspeech2](https://arxiv.org/abs/1512.02595) model in `pytorch`. Getting this to work required some efforts because of using `warp-ctc` bindings from Baidu. These are highly GPU and CPU optimized operations for calculating CTC loss that is used in both models.

To get that to work I had to also compile pytorch from sources as it was built against `libgomp` of other version and `import warp_ctc` command failed with

    path-to-pytorch-included-libs/libgomp.so.1: version `GOMP_4.0' not found (required by path-to-built-warpctc/libwarpctc.so.0)

Obviously, `libwarpctc` was linked against `libgomp` of a different version than a pytorch binary libraries (there are a bunch of libs in pytorch distribution). To fix that I had to built pytorch from sources as well (to link it against the same lib). So the whole path I need to take was:

 1. Install `torch` (from sources, required by `warp-ctc` library)
 1. Install `pytorch` (from sources)
 1. Compile and install `warp-ctc`
 1. Profit, finally.

Original implementation uses `an4` dataset for model training. Dataset is not big and training takes about an hour with default settings.

However, I was mainly interested in training the model with VCTK (as a larger dataset I have already used). The implementation provides simple instruction how to convert the dataset to train: just provide the manifest file (that is, simple csv with path to `wav` file and path to the corresponding transcription). It took me a while to preprocess dataset to comply with the model. The overall process is as follows:

 1. Download VCTK
 1. Downsample it to 16k (or change default settings and model definition a little)
 1. Create manifest file
 1. Modify source to properly parse transcription text files

Following snippet uses `zsh` globbing feature to download and preprocess VCTK dataset:

```sh
mkdir dataset/; cd dataset/
wget http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz
tar xfz VCTK-Corpus.tar.gz
cd ..

for file in dataset/wav48/**/*.wav; do ffmpeg -i $file -ac 1 -ar 16000 -acodec pcm_s16le "${file/wav48/wav16}" ; done
```

Create manifests with `ipython`:

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

Unfortunately, there is no transcription for `p315` entries. We remove those from the train and test sets.

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
+        try:
+            transcript = [self.labels_map[x.upper()] for x in transcript]
+        except KeyError as e:
+            print(e, transcript)
+            raise
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

Training with default parameters fails for me because of exploding gradients. I have to reduce learning rate to start it going. I didn't perform any (grid/random) search, just decreased it a little.

    python train.py --train_manifest train_m.csv --val_manifest test_m.csv --epochs 200 --lr 1e-5

Unfortunately, currently implementation (with my changes) suffers from a memory leak problem. Using 32Gb RAM machine doesn't allow me to train the model for longer than 10 epochs. To quickly overcome this difficulty (without spending time on debugging real issue) I just added feature to resume training from previous checkpoint. Here  is diff that needs to be applied:

```diff
diff --git a/train.py b/train.py
index 81e0488..ddb5d0b 100644
--- a/train.py
+++ b/train.py
@@ -37,6 +37,8 @@ parser.add_argument('--epoch_save', default=False, type=bool, help='Save model e
 parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
 parser.add_argument('--final_model_path', default='models/deepspeech_final.pth.tar',
                     help='Location to save final model')
+parser.add_argument('--resume', default='',
+                    help='Restore model before training: resume training')


 class AverageMeter(object):
@@ -100,8 +102,15 @@ def main():

     model = DeepSpeech(rnn_hidden_size=args.hidden_size, nb_layers=args.hidden_layers, num_classes=len(labels))
     decoder = ArgMaxDecoder(labels)
+
     if args.cuda:
         model = torch.nn.DataParallel(model).cuda()
+
+    if args.resume:
+        package = torch.load(args.resume)
+        model.load_state_dict(package['state_dict'])
+        model.eval()
+
     print(model)
     parameters = model.parameters()
     optimizer = torch.optim.SGD(parameters, lr=args.lr,
```

Training takes several hours (3 or 4, don't have the exact number) for 10 epochs on one NVIDIA Titan X Pascal GPU. I trained it for 50 epochs. It achieves WER of 39 and CER of 13. For testing purposes, I used exactly the same audio files as for the previous model. Here are the results so far:

    $ python predict.py --model_path models/deepspeech_10.pth.tar --audio_path ~/vcs/speech-to-text-wavenet/asset/data/wav48/p225/p225_001.wav
    POLEAIS QKARTVILSD STENBAYF

    $ python predict.py --model_path models/deepspeech_10.pth.tar --audio_path ~/vcs/speech-to-text-wavenet/asset/data/wav48/p236/p236_005.wav
    SHE CAN SCOOPE THES THINGS INTO THREE RED BAGS AND WE WILL GO MEET  HER WENES DAY AT THE TRAINS TATIONUBUHW

    $ python predict.py --model_path models/deepspeech_10.pth.tar --audio_path ~/vcs/speech-to-text-wavenet/asset/data/wav48/p317/p317_063.wav
    I WONDERED ABOUT CHANGING BACK TO REUGBY

Wow! Results are quite interesting (given I tried only three samples from _training_ set). First one doesn't look good at all (the previous model did worse on the first sample also), but two others are quite good. My guess is a proper implementation (that will allow longer training) and some hyperparameter tuning could further improve the results.



## Audio synthesis

For synthesis, I took SampleRNN implementation from paper authors. Model training time is 72 hours per experiment (with default running options). The first experiment used classic piano music (Bach dataset). Results are present below.

24 hours of training, 9 epochs:

<audio src="{{site.url}}/assets/samplernn/sample_e9_i71363_t24.00_tr1.0442_v1.1383_best_17.wav" controls="controls"></audio>

72 hours of training, 28 epochs:

<audio src="{{site.url}}/assets/samplernn/sample_e28_i205912_t72.00_tr0.9759_v1.0677_best_14.wav" controls="controls"></audio>

For me, it sounds really nice! Results are already good enough even after 1 full day of training but some noise generation is possible. After 3 days of training, noise being generated possibility is much smaller (but it still can occur).

Next experiment is to use VCTK speech and let the algorithm do the magic. Note that model is unconditional so it won't learn how to properly spell the words. Rather we can expect it to be some _'alien'_ language that slightly resembles English.

24 hours of training, 1 epoch:

<audio src="{{site.url}}/assets/samplernn/sample_e1_i67879_t24.00_tr1.0129_v1.0562_best_11.wav" controls="controls"></audio>

72 hours of training, 5 epochs:

<audio src="{{site.url}}/assets/samplernn/sample_e5_i204624_t72.00_tr1.1449_v1.0247_best_17.wav" controls="controls"></audio>

These are impressive! I understand nothing they pronounce but my English could be not that good :) Dataset is bigger than piano music so it takes more time for 1 epoch. Also, speech and music are different but the model does pretty good job in term of naturalness.

## Conclusion

Current state-of-the-art results are very promising. Given large datasets and enough computation power, one can build pretty interesting solutions. I definitely look forward to [DeepVoice][deepvoice] source code to be released. Meanwhile, I better spend time collecting dataset big and diverse enough to produce awesome results.

[deepvoice]: https://arxiv.org/abs/1702.07825
[wavenet-asr]: https://github.com/buriburisuri/speech-to-text-wavenet
[deepspeech2-asr]: https://github.com/SeanNaren/deepspeech.pytorch
