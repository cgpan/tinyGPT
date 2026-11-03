"""Character-level tokenizer: turn text into a list of integers and back."""


class CharTokenizer:
    """Gives every distinct character its own number.

    >>> tok = CharTokenizer.from_text("hello")
    >>> tok.chars
    ['e', 'h', 'l', 'o']
    >>> tok.encode("hell")
    [1, 0, 2, 2]
    >>> tok.decode([1, 0, 2, 2])
    'hell'
    """

    def __init__(self, chars):
        chars = list(chars)
        if len(set(chars)) != len(chars):
            raise ValueError("the vocabulary contains a character twice")
        self.chars = chars                                  # number -> character
        self.stoi = {ch: i for i, ch in enumerate(chars)}   # character -> number

    @classmethod
    def from_text(cls, text):
        """Build the vocabulary from every distinct character in `text`, in sorted order."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        """Text -> list of integers."""
        ids = []
        for ch in text:
            if ch not in self.stoi:
                raise ValueError(f"{ch!r} is not in the vocabulary, so the model can never read or write it")
            ids.append(self.stoi[ch])
        return ids

    def decode(self, ids):
        """List of integers (or a tensor or array of them) -> text."""
        return "".join(self.chars[int(i)] for i in ids)
