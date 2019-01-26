# Review

```link
http://ruder.io/deep-learning-nlp-best-practices/
```

## Embedding dimensionality

```text
2048-dimensional embeddings yield the best performance, but only do so by a small margin.
Even 128-dimensional embeddings perform surprisingly well and converge almost twice as quickly (Britz et al., 2017).
```

## Layers Number

```text
The encoder does not need to be deeper than 2−4layers.
Deeper models outperform shallower ones, but more than 4 layers is not necessary for the decoder (Britz et al., 2017).
```

## Directionality

```text
Sutskever et al., (2014) [64] proposed to reverse the source sequence to reduce the number of long-term dependencies.
Reversing the source sequence in unidirectional encoders outperforms its non-reversed counter-part (Britz et al., 2017).
```

## Beam search

```text
https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
Medium beam sizes around 10 with length normalization penalty of 1.0 (Wu et al., 2016) yield the best performance (Britz et al., 2017).
```

## Attention Visualize

https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb?fbclid=IwAR24r5o20ewCVpXm0IZ4yGQ-v_lfAqQRJEtHErUC2eCS0YI4D2VBj3HsVeM

## Bleu score

```link
    - https://slator.com/technology/how-bleu-measures-translation-and-why-it-matters/
    - https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
```

- calculate with bleu.py
- Maximum n-gram order = 4 with smooth=False
- nmt bleu score was x100 as 100%

## Decay