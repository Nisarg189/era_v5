"""
Kronecker byte codec, and its exact inverse.

Forward codec (faithful to the Session 7 lesson specification):

    kappa(token) = znorm( (1/sqrt(L)) * vec( sum_p  c[byte_p] (x) p[position_p] ) )

    - a grid of char_dim rows (one per byte value, 0..255) and pos_dim columns
      (one per byte position, 0..31),
    - L = min(len(utf8_bytes), pos_dim); bytes past position pos_dim are dropped,
    - each byte b at position p marks cell (b, p) with 1,
    - the grid is flattened to a vector of length char_dim * pos_dim = 8192,
    - scaled by 1/sqrt(L) so short and long tokens come out at comparable scale,
    - then z-normalised (subtract mean, divide by standard deviation).

The codec holds no trainable parameters. It is a fixed function of the token bytes.

The inverse, decode(), is exact for any token whose byte length is <= pos_dim.
Two tokens that agree on their first pos_dim bytes produce the same code and are
therefore not separable: that is the collision the byte-window audit measures, and
it is the only way this codec loses information.
"""

import numpy as np

CHAR_DIM = 256
POS_DIM = 32
CODE_DIM = CHAR_DIM * POS_DIM  # 8192


class KroneckerCodec:
    def __init__(self, char_dim: int = CHAR_DIM, pos_dim: int = POS_DIM):
        self.char_dim = char_dim
        self.pos_dim = pos_dim
        self.code_dim = char_dim * pos_dim

    # ---- forward ---------------------------------------------------------

    def cells(self, token: str):
        """The (byte_value, position) cells a token marks, in position order."""
        b = token.encode("utf-8")
        L = min(len(b), self.pos_dim)
        return [(b[p], p) for p in range(L)]

    def encode(self, token: str) -> np.ndarray:
        b = token.encode("utf-8")
        L = min(len(b), self.pos_dim)
        grid = np.zeros((self.char_dim, self.pos_dim), dtype=np.float64)
        if L == 0:
            # empty token: return the z-norm of an all-zero grid, which is all zero
            return grid.reshape(-1)
        for p in range(L):
            grid[b[p], p] = 1.0
        vec = grid.reshape(-1) / np.sqrt(L)
        mean = vec.mean()
        std = vec.std()
        if std == 0:
            return vec - mean
        return (vec - mean) / std

    def encode_batch(self, tokens) -> np.ndarray:
        out = np.zeros((len(tokens), self.code_dim), dtype=np.float64)
        for i, t in enumerate(tokens):
            out[i] = self.encode(t)
        return out

    # ---- exact inverse ---------------------------------------------------

    def decode(self, code: np.ndarray) -> str:
        """Recover the token string from an exact code.

        After 1/sqrt(L) scaling and z-normalisation the marked cells all share one
        (high) value and the unmarked cells share another (low) value, so the marked
        set is exactly the cells above the midpoint. Each marked cell names one byte
        at one position; sorting by position rebuilds the byte string.
        """
        grid = code.reshape(self.char_dim, self.pos_dim)
        hi, lo = grid.max(), grid.min()
        if hi == lo:
            return ""  # all-equal grid carries no marks (empty token)
        thresh = (hi + lo) / 2.0
        marked = np.argwhere(grid > thresh)  # rows of (byte_value, position)
        by_pos = {}
        for value, pos in marked:
            by_pos[int(pos)] = int(value)
        b = bytes(by_pos[p] for p in sorted(by_pos))
        return b.decode("utf-8", errors="replace")

    def roundtrip_ok(self, token: str) -> bool:
        """True when the token survives encode->decode unchanged.

        A token longer than pos_dim bytes is truncated by the window, so only its
        first pos_dim bytes can return; the comparison is against that truncation,
        which is the most the codec can preserve.
        """
        b = token.encode("utf-8")
        recoverable = b[: self.pos_dim].decode("utf-8", errors="replace")
        return self.decode(self.encode(token)) == recoverable
