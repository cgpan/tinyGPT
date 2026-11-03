# Project 8 · Notes on the solution

Try the project yourself first. A solution you have read teaches much less than one you
fought with, even if you didn't finish.

The code is in [`scratchgpt/watermark.py`](scratchgpt/watermark.py).

## `detect`

For each token after the first, check whether it is in the green list that its previous
token picks. With T tokens checked and G green, unmarked text has G ≈ γT with variance
Tγ(1 − γ), so z = (G − γT) / √(Tγ(1 − γ)). With γ = 0.25 and 150 tokens, chance gives
about 37 green tokens with a standard deviation of 5.3; z = 4 means about 59.

## Predict 2 and 4

- **Random replacement.** Each replaced token is green by chance, and it also changes the
  green list of the token after it. A check survives only if both its tokens do
  (0.75 × 0.75 when a quarter are replaced), so a quarter spoils nearly half the checks.
  With `base`, 20 of 20 texts (19 of 20 on the Mac) were still flagged after a quarter of
  their tokens were replaced, and 0 of 20 (1 of 20) after half.
- **False alarms.** A real review is green at the chance rate, so z > 4 happens about 3 times
  in 100,000. In 20 real reviews, expect none.

## What we measured with `base`

| 20 texts each | mean z | flagged (z > 4) |
|---|---|---|
| watermarked | 11.3 (PC), 11.1 (Mac) | 100% |
| not watermarked | 0.2, −0.3 | 0% |
| real test reviews | −0.3 | 0% |

## Predict 3: low-entropy text

When the model is nearly sure of the next token, adding 2 to the green scores rarely
changes the choice, so that token carries almost no watermark. Text that is predictable
(repeated phrases, a stock opening) is weakly marked; the watermark lives in the tokens
where the model had real choice. It shows, roughly, in the numbers: per token checked,
`base`'s watermark (z 11.1 to 11.3 over 149 tokens) is about three quarters as strong as
that of a tiny, much less certain test model (z 12.6 over 99 tokens), though the two also
differ in data and vocabulary.

## Investigate 3: the key

Anyone with the key can detect the watermark, and also forge it: write text that is all
green, so a person's text is "detected" as generated, or remove it carefully. That's why
the key is kept private, and why a public detection service is safer than a public key.
