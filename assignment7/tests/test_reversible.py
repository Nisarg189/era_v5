"""
Invariant tests for the reversible-Kronecker claim. Offline, no training, under a
second. Run: python3 -m tests.test_reversible
"""
import numpy as np
from rkron.codec import KroneckerCodec, CODE_DIM
from rkron.model import TinyLM


def test_codec_roundtrip_in_window():
    c = KroneckerCodec()
    for t in ["the", "training", "a", "भारत", "9", "తెలుగు", "ગુજરાત", "x_1", "def"]:
        assert c.roundtrip_ok(t), f"round-trip failed for {t!r}"
    print("ok  codec round-trip is exact for in-window tokens")


def test_collision_then_separation():
    w1, w2 = "अंतर्राष्ट्रीयकरण", "अंतर्राष्ट्रीयता"   # >32 bytes, share first 32
    c32, c64 = KroneckerCodec(pos_dim=32), KroneckerCodec(pos_dim=64)
    assert np.allclose(c32.encode(w1), c32.encode(w2)), "expected collision at 32"
    assert not np.allclose(c64.encode(w1), c64.encode(w2)), "expected separation at 64"
    print("ok  known Hindi pair collides at pos_dim=32 and separates at 64")


def test_head_param_accounting():
    toks = ["a", "b", "c", "the", "of", "भारत"]
    dense = TinyLM(toks, d_model=32, block_size=8, n_layer=1, input_path="kron", head="dense")
    tied = TinyLM(toks, d_model=32, block_size=8, n_layer=1, input_path="kron", head="tied_kron")
    assert dense.head_param_count() == len(toks) * 32
    assert tied.head_param_count() == 0, "tied head must add no V-dependent parameters"
    print("ok  tied head has zero vocabulary-dependent parameters; dense head has V*D")


def test_adjoint_identity():
    """The tied head's logits equal q @ C^T exactly: the head is the codec adjoint,
    not a stored V x D matrix."""
    toks = ["a", "b", "the", "of", "भारत", "def", "9"]
    m = TinyLM(toks, d_model=48, block_size=6, n_layer=1, input_path="kron",
               head="tied_kron", seed=1)
    x = np.array([[0, 1, 2, 3, 4, 5]])
    logits, H = m.forward(x)
    q = H.data @ m.W_in.data.T                       # [B,T,8192]
    via_code = (q @ m.C.data.T) * m.logit_scale.data
    assert m.C.data.shape[1] == CODE_DIM
    diff = float(np.abs(logits.data - via_code).max())
    assert diff < 1e-9, f"adjoint identity broken, max diff {diff}"
    print(f"ok  logits == q @ C^T  (max diff {diff:.1e}); head reuses only the shared W_in")


def test_argmax_equals_nearest_code():
    """Predicting the next token = choosing the nearest code = a decode, with no head
    matrix and no fixed vocabulary size."""
    toks = ["a", "b", "the", "of", "भारत", "def", "9", "to", "in"]
    m = TinyLM(toks, d_model=48, block_size=5, n_layer=1, input_path="kron",
               head="tied_kron", seed=2)
    x = np.array([[0, 1, 2, 3, 4]])
    logits, H = m.forward(x)
    argmax = logits.data[0, -1].argmax()
    q = H.data[0, -1] @ m.W_in.data.T
    codes = m.C.data                                 # generated from bytes
    nearest = (codes @ q).argmax()                   # nearest code by dot product
    assert argmax == nearest
    print("ok  argmax logit == nearest code (prediction is a codec inversion)")


def main():
    for fn in [test_codec_roundtrip_in_window, test_collision_then_separation,
               test_head_param_accounting, test_adjoint_identity,
               test_argmax_equals_nearest_code]:
        fn()
    print("ALL REVERSIBLE-KRONECKER INVARIANTS PASS")


if __name__ == "__main__":
    main()
