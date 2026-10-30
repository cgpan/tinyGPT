"""Project 1: clean, scrub, deduplicate, decontaminate, split."""

import numpy as np
import pytest
from conftest import long_review

from scratchgpt import data, imdb


# --- clean -----------------------------------------------------------------------------

def test_clean_turns_html_line_breaks_into_newlines():
    assert data.clean("One.<br />Two.<br/>Three.<BR>Four.") == "One.\nTwo.\nThree.\nFour."


def test_clean_keeps_at_most_one_blank_line():
    assert data.clean("One.<br /><br /><br /><br />Two.") == "One.\n\nTwo."


def test_clean_decodes_html_entities():
    assert data.clean("Tom &amp; Jerry &quot;rule&quot; &#39;99") == "Tom & Jerry \"rule\" '99"


def test_clean_collapses_spaces_and_strips_lines():
    assert data.clean("  Too   many \t spaces  <br />  here.  ") == "Too many spaces\nhere."


def test_clean_turns_non_breaking_spaces_into_spaces():
    assert data.clean("a&nbsp;b c") == "a b c"


def test_clean_normalizes_unicode():
    assert data.clean("Café") == "Café"            # e + combining accent -> é


# --- scrub -----------------------------------------------------------------------------

def test_scrub_replaces_emails():
    text, counts = data.scrub("Mail me at jane.doe+imdb@mail.example.co.uk today.")
    assert text == "Mail me at [email] today."
    assert counts == {"email": 1, "url": 0, "phone": 0}


def test_scrub_replaces_web_addresses():
    text, counts = data.scrub("See http://example.com/a?b=1 and www.example.org.")
    assert text == "See [url] and [url]"
    assert counts["url"] == 2


@pytest.mark.parametrize("phone", ["555-123-4567", "(555) 123-4567", "555.123.4567", "+1 555 123 4567"])
def test_scrub_replaces_phone_numbers(phone):
    text, counts = data.scrub(f"Call {phone} now.")
    assert text == "Call [phone] now."
    assert counts["phone"] == 1


@pytest.mark.parametrize("text", ["Rated 7/10.", "Released in 1999-2000.", "It ran 120 minutes.",
                                  "Top 250 list, 2nd place.", "Episode 4: A New Hope (1977)"])
def test_scrub_leaves_other_numbers_alone(text):
    assert data.scrub(text) == (text, {"email": 0, "url": 0, "phone": 0})


# --- exact and near duplicates --------------------------------------------------------

def test_fingerprint_ignores_case_whitespace_and_punctuation():
    assert data.fingerprint("Great film!  Loved it.") == data.fingerprint("great film loved it")
    assert data.fingerprint("Great film!") != data.fingerprint("Great films!")


def test_fingerprint_is_a_sha1_hex_digest():
    fp = data.fingerprint("anything")
    assert len(fp) == 40 and int(fp, 16) >= 0


def test_shingles_are_lower_case_word_ngrams():
    assert data.shingles("The cat sat on the mat", n=3) == {"the cat sat", "cat sat on", "sat on the", "on the mat"}
    assert data.shingles("Too short", n=5) == {"too short"}
    assert data.shingles("!!!", n=5) == set()


def test_minhash_is_deterministic_and_has_the_right_shape():
    s = data.shingles(long_review(3))
    a, b = data.minhash(s, num_hashes=64), data.minhash(set(s), num_hashes=64)
    assert a.shape == (64,) and a.dtype == np.uint64
    assert np.array_equal(a, b)
    assert np.all(data.minhash(set(), num_hashes=8) == data.PRIME)


def test_minhash_estimates_jaccard_similarity():
    words = [f"w{i}" for i in range(400)]
    a = set(words[:300])
    b = set(words[100:400])                                   # true Jaccard: 200 / 400 = 0.5
    estimate = data.estimated_jaccard(data.minhash(a, 256), data.minhash(b, 256))
    assert abs(estimate - 0.5) < 0.1
    assert data.estimated_jaccard(data.minhash(a, 256), data.minhash(a, 256)) == 1.0


def test_candidate_pairs_find_similar_documents_and_skip_different_ones():
    texts = [long_review(10), long_review(10).replace("the", "a", 1), long_review(11), long_review(12)]
    signatures = [data.minhash(data.shingles(t)) for t in texts]
    pairs = data.candidate_pairs(signatures, bands=16)
    assert (0, 1) in pairs
    assert all(i < j for i, j in pairs)
    assert (2, 3) not in pairs


def test_near_duplicates_reports_pairs_above_the_threshold():
    texts = [long_review(20), long_review(21), long_review(20).replace("the", "a", 1)]
    found = data.near_duplicates(texts, threshold=0.8)
    assert [(i, j) for i, j, _ in found] == [(0, 2)]
    assert 0.8 <= found[0][2] <= 1.0


# --- the given reader -----------------------------------------------------------------

def test_read_reviews_reads_every_folder(imdb_root):
    reviews = list(imdb.read_reviews(imdb_root))
    assert len(reviews) == 10
    first = reviews[0]
    assert first["id"] == "train/pos/0" and first["split"] == "train"
    assert first["label"] == "pos" and first["rating"] == 9 and first["movie"] == "tt1000000"
    unsup = [r for r in reviews if r["split"] == "unsup"]
    assert len(unsup) == 2 and all(r["label"] is None and r["rating"] is None for r in unsup)


# --- the whole pipeline ---------------------------------------------------------------

@pytest.fixture
def splits(imdb_root):
    return data.build_splits(imdb.read_reviews(imdb_root), val_fraction=0.25, seed=0)


def test_build_splits_counts_what_it_did(splits):
    _, stats = splits
    assert stats["reviews"] == 10
    assert stats["exact_duplicates"] == 1
    assert stats["near_duplicates"] == 1
    assert stats["contaminated"] == 1
    assert (stats["email"], stats["url"], stats["phone"]) == (1, 2, 1)
    assert (stats["train"], stats["val"], stats["test"]) == (3, 1, 3)


def test_build_splits_keeps_the_first_of_each_duplicate(splits):
    split, _ = splits
    assert [d["id"] for d in split["train"]] == ["train/pos/0", "train/pos/1", "train/neg/0"]


def test_build_splits_never_trains_on_held_out_text(splits):
    split, _ = splits
    held_out = {data.fingerprint(d["text"]) for d in split["val"] + split["test"]}
    assert not any(data.fingerprint(d["text"]) in held_out for d in split["train"])
    assert all(d["id"].startswith("test/") for d in split["val"] + split["test"])


def test_build_splits_cleans_scrubs_and_keeps_labels(splits):
    split, _ = splits
    by_id = {d["id"]: d for docs in split.values() for d in docs}
    assert by_id["train/pos/0"]["text"] == "A wonderful film.\n\nThe acting was great & the music was lovely."
    assert "[email]" in by_id["train/neg/0"]["text"] and "[phone]" in by_id["train/neg/0"]["text"]
    assert by_id["train/pos/0"]["label"] == "pos"


def test_build_splits_shuffles_held_out_reviews_with_the_seed(imdb_root):
    a, _ = data.build_splits(imdb.read_reviews(imdb_root), val_fraction=0.25, seed=0)
    b, _ = data.build_splits(imdb.read_reviews(imdb_root), val_fraction=0.25, seed=0)
    assert a == b
    order = np.random.default_rng(0).permutation(4)
    test_ids = ["test/pos/0", "test/pos/1", "test/neg/0", "test/neg/1"]
    assert a["val"][0]["id"] == test_ids[order[0]]
