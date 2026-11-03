# Credits

tinyGPT builds on other people's work. This page lists what comes from where, and under
which terms.

## Material from other projects

| what | from | terms | how it is used here |
|---|---|---|---|
| `entry/src/tinygpt/shakespeare.txt` (tinyshakespeare, 1.1 MB) | Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn), `data/tinyshakespeare` | char-rnn is MIT-licensed, © Andrej Karpathy; Shakespeare's plays are in the public domain | copied into this repository as the entry track's corpus |
| GPT-2 small, in entry chapter 8 | OpenAI (Radford et al., 2019), from Hugging Face: [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) | MIT on Hugging Face; OpenAI's [gpt-2](https://github.com/openai/gpt-2) repository uses a modified MIT license | downloaded when the chapter runs; not part of this repository |
| GPT-2's pre-tokenization pattern (`SPLIT_PATTERN` in the advanced track's `tokenizer.py`) | OpenAI's [gpt-2](https://github.com/openai/gpt-2), `src/encoder.py` | as above | rewritten for Python's `re` module |
| The IMDB Large Movie Review Dataset | Maas et al. (2011), [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/) | its authors' terms; they ask that you cite their paper | downloaded by `advanced/scripts/prepare_data.py`; not part of this repository |
| The "Open in Colab" badge | [Google Colab](https://colab.research.google.com) | Google's | linked, not copied |

## Designs this project follows

Its code was written for tinyGPT, following these designs:

- **Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)** (MIT) and his lecture
  *Let's build GPT: from scratch, in code, spelled out*, with its code in
  [ng-video-lecture](https://github.com/karpathy/ng-video-lecture). The entry track's model
  (heads of attention, pre-norm blocks, and `base`'s size: 6 layers, 6 heads, width 384,
  dropout 0.2) and its path from a bigram model to a transformer follow them.
- **Stanford's [CS336, Language Modeling from Scratch](https://stanford-cs336.github.io/)**:
  the advanced track's format, a test suite that specifies every piece students write. Its
  projects, tests and code are our own.
- **PRIMM** (Sentance, Waite and Kallia, 2019): the Predict, Run, Investigate, Modify, Make
  steps of every entry chapter, and the Predict step of every advanced project.

## Software

Installed by `pip`, not included here: [PyTorch](https://pytorch.org) (BSD-3-Clause),
[NumPy](https://numpy.org) (BSD-3-Clause), [PyYAML](https://pyyaml.org) (MIT),
[pytest](https://pytest.org) (MIT), [Matplotlib](https://matplotlib.org) (Matplotlib license),
Jupyter's nbformat, nbclient and ipykernel (BSD-3-Clause), and, for entry chapter 8 only,
Hugging Face's [transformers](https://github.com/huggingface/transformers) and tokenizers
(Apache 2.0).

## The ideas behind the advanced track

Each project builds a method from one of these papers. You don't need to read them to do
the projects; they are where to go further.

1. **Data.** Broder (1997), *On the resemblance and containment of documents* (MinHash).
   Leskovec, Rajaraman and Ullman, *Mining of Massive Datasets*, chapter 3 (locality-sensitive
   hashing). Lee et al. (2022), *Deduplicating Training Data Makes Language Models Better*.
   Gebru et al. (2021), *Datasheets for Datasets*.
2. **Tokenizer.** Sennrich, Haddow and Birch (2016), *Neural Machine Translation of Rare Words
   with Subword Units* (BPE). Radford et al. (2019), *Language Models are Unsupervised
   Multitask Learners* (GPT-2's byte-level BPE).
3. **Model.** Vaswani et al. (2017), *Attention Is All You Need*. Zhang and Sennrich (2019),
   *Root Mean Square Layer Normalization*. Su et al. (2021), *RoFormer: Enhanced Transformer
   with Rotary Position Embedding*. Shazeer (2020), *GLU Variants Improve Transformer*
   (SwiGLU). Press and Wolf (2017), *Using the Output Embedding to Improve Language Models*.
4. **Training.** Loshchilov and Hutter (2019), *Decoupled Weight Decay Regularization*
   (AdamW), and (2017), *SGDR: Stochastic Gradient Descent with Warm Restarts* (the cosine
   schedule). Micikevicius et al. (2018), *Mixed Precision Training*. Holtzman et al. (2020),
   *The Curious Case of Neural Text Degeneration* (top-p sampling).
5. **Performance.** Kaplan et al. (2020), *Scaling Laws for Neural Language Models* (FLOPs per
   token). Chowdhery et al. (2022), *PaLM: Scaling Language Modeling with Pathways* (MFU).
   Korthikanti et al. (2022), *Reducing Activation Recomputation in Large Transformer Models*
   (activation memory). Dao et al. (2022), *FlashAttention: Fast and Memory-Efficient Exact
   Attention with IO-Awareness*.
6. **Scaling.** Kaplan et al. (2020), as above. Hoffmann et al. (2022), *Training
   Compute-Optimal Large Language Models*. Muennighoff et al. (2023), *Scaling
   Data-Constrained Language Models*. Penedo et al. (2024), *The FineWeb Datasets* (for the
   stretch goal).
7. **Evaluation.** Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*. Gao et
   al. (2020), *The Pile*, which compares models in bits per byte. Alain and Bengio (2016),
   *Understanding intermediate layers using linear classifier probes*. Carlini et al. (2021),
   *Extracting Training Data from Large Language Models*, and (2023), *Quantifying
   Memorization Across Neural Language Models*.
8. **Release.** Kirchenbauer et al. (2023), *A Watermark for Large Language Models*. Mitchell
   et al. (2019), *Model Cards for Model Reporting*. US Federal Trade Commission (2024),
   *Trade Regulation Rule on the Use of Consumer Reviews and Testimonials*.

For the entry track's teaching approach: Sentance, Waite and Kallia (2019), *Teaching
computer programming with PRIMM: a sociocultural perspective*, Computer Science Education.
