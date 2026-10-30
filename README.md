# tinyGPT

Learn how a language model works by building one. There are two tracks; start with the
one that fits you.

| | [Entry track](entry/README.md) | [Advanced track](advanced/README.md) |
|---|---|---|
| **You** | are curious how ChatGPT-like models work | want to build language models for a living |
| **You do** | run, read and change a working model, one piece per chapter | write every piece yourself, against a test suite |
| **The model** | a character-level GPT, 10.7M parameters | a word-piece GPT with a tokenizer you trained, sized to your hardware |
| **The data** | Shakespeare, 1 MB, included | 100,000 IMDB movie reviews, 133 MB, downloaded by the setup |
| **Time** | 8 chapters, a few hours each | 8 projects, several weeks |
| **Hardware** | Colab's free GPU, any NVIDIA gaming GPU, or an Apple Silicon Mac | an NVIDIA GPU with 12 GB or more, a Mac with 24 GB or more, or paid Colab |
| **Longest run** | under 10 minutes | the final run can take up to 24 hours |

The entry track is a good start for the advanced one. Its last chapter, on comparing
models with different tokenizers, is where the advanced track begins.

## Responsible use

This project is for learning how language models work. **Never use it, or a model you
train with it, to write fake reviews or comments, to pose as real people, or to spread
misinformation.**

The reviews the advanced track learns from were written by real people; what a model
writes was not. Passing generated text off as a real review deceives the people who read
it, and in the United States it can be illegal: the FTC's rule on fake reviews, in force
since October 2024, covers reviews by people who don't exist, including AI-generated
ones, and allows fines for each one.

The advanced track builds three habits around this:

- Text written by its sampler carries a hidden watermark by default, and project 8 has
  you build the detector that finds it.
- Nothing in it steers the writing towards a star rating, a product or a film, and
  there is no script for writing reviews in bulk.
- Every trained model gets a model card that says what it must not be used for.

None of this stops someone determined to misuse it: the code is yours to change. These
are the habits expected of the people who build language models.

## License and data

The code is under the [MIT license](LICENSE). Shakespeare's plays are in the public
domain. The IMDB reviews are not part of this repository: the advanced track downloads
them from Stanford, and they remain under their authors' terms. If you publish results
from them, cite Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*.

## Credits

The entry track's model follows Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) and
[ng-video-lecture](https://github.com/karpathy/ng-video-lecture). The advanced track's
format, a test suite that specifies every piece students write, follows Stanford's
[CS336, Language Modeling from Scratch](https://stanford-cs336.github.io/); its projects
and code are our own.
