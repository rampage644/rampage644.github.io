---

title: Wavenet
---

This is 3rd post in a series of post about my way to implement [Wavenet](wavenet-paper). In first two posts ([one]({{ site.baseurl }}{% post_url 2017-2-22-pixelcnn %}
), [two]({{ site.baseurl }}{% post_url 2017-2-24-gated-pixelcnn %}
)) I talked about implementing Gated PixelCNN that I considered prerequisite for myself before starting coding and training Wavenet.

As PixelCNN (kind of) successfully implemented and I quite content with the results I'm diving into Wavenet implementation. As a side note, I found WaveNet architecture to be much simplier and implementation more straighforward that those of PixelCNN's. I think this is partially because I devoted much time to understand many issues while working on PixelCNN. Another idea is that 1D audio signal a little simplier that 2D image to work with.

## Dilated convolutions

First thing I faced were dilated convolutions. [This paper](https://arxiv.org/pdf/1511.07122.pdf) have excellent overall description of what it is and why do we need them. My own explanation follows.

First of all, main difference between dilated convolutions and regular ones are what inputs get combined to produce output. Dilation assumes you're skipping some of them:

![Dilated convolutions]({{site.url}}/assets/2017-02-23-155353_1250x430_scrot.png)

Here we perform convolution with kernel size of 3 (`3x3`). First image show that all neighbor pixel participate in calculating output. Second image takes every 2nd input (that is, dilate input) and we speak about dilation rate (or factor) of 2. Third image takes every 4th, dilation rate is 4. Regular convolutions take every 1st (all) inputs so dilation rate is 1.

One can think about dilated convolutions as expanding filter with zeros interspersing actual weights.

![Dilation filter]({{site.url}}/assets/dilation_filter.png)

The reason we need such convolutions is increasing receptive field size. ![dilation stack]({{site.url}}/assets/2017-02-23-155956_803x294_scrot.png) Consider the very last output in top layer (one with dilation = `8`). It has access to (receptive field size equal to) 16 samples. Had we hadn't use dilations we'll end up with only `5`. 8 and 5 are not that big to notice and appreciate the difference but with dilation we get exponential growth rate. instead of linear.

For example, here is formula you can use to calculate receptive field size (function of layers number and stack number): $$ size = stack\_num * (2 ^ {layers\_num} - 1) $$. Please note, it's only valid for stride = 1, kernel width = 2.

## $$ \mu $$-law

Authors use categorical distribution of output amplitudes as was done in PixelCNN. Input data is 16bit PCM, so output softmax would end up having 65536 categories. In order to decrease computational cost $$ \mu $$-law encoding was used [wiki](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm). This transformation takes input in `[-1; 1]` range and produces output within exactly same range but slightly changes distribution making it more uniform.

`(figures of audio sample data distribution)`

## Architecture

![Arcitecture]({{site.url}}/assets/2017-02-23-161420_692x358_scrot.png)

The overall architecture is quite simple (given you have encountered it in PixelCNN):
 * First layer mask 'A' causal convoluition
 * Several stack of layers with dilated convolutions and gated blocks
 * Skip connections
 * Relu, 1x1 conv, relu, 1x1 conv, softmax

Something I want to emphasize here: **Don't forget** about causality in the very first layer! You have to break information flow from sample being predicted and your network. Again, [Reed](reed-paper) helps here as well (note $$ x_t $$ and $$ \hat{x_t} $$ aren't connected)

 ![1st causal layer]({{site.url}}/assets/2017-02-23-161845_391x344_scrot.png)

## Data preparation

For training you will need some audio data (or any 1D signal, for example stocks or whatever).

The trick to remember is that output signal depends on previous `receptive_field_size` samples, so while calculating you should be aware of that. For example, the very first predicted sample would be calculated using zeros as input (because of padding with zeros in all layers). Idea here is to calculate loss only for those samples that are conditioned of already visible signal. Here is visual explaination.

`(image of loss and receptive_field_size)`

My experiments show (I infored that at first) that it will help model to traing bu during inference it will take a model some time to stop producing noise and cracking.

## References

[wavenet-blog]: https://deepmind.com/blog/wavenet-generative-model-raw-audio/
[wavenet-paper]: https://arxiv.org/pdf/1609.03499.pdf
[pixelrnn-paper]: https://arxiv.org/pdf/1601.06759.pdf
[pixelcnn-paper]: https://arxiv.org/pdf/1606.05328.pdf
[reed-paper]: http://www.scottreed.info/files/iclr2017.pdf
[made-paper]: https://arxiv.org/pdf/1502.03509.pdf
[repo]: https://github.com/rampage644/wavenet
