from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RepresentationSlices:
  z: np.ndarray
  prefixes: tuple
  blocks: tuple
  prefix_dims: tuple
  block_dims: tuple


def validate_prefix_dims(prefix_dims, z_dim=None):
  dims = tuple(int(x) for x in prefix_dims)
  if not dims or any(a >= b for a, b in zip(dims[:-1], dims[1:])):
    raise ValueError(f'Prefix dimensions must be strictly increasing: {dims}')
  if z_dim is not None and dims[-1] != int(z_dim):
    raise ValueError(f'Last prefix {dims[-1]} does not equal z dim {z_dim}')
  return dims


def block_dims(prefix_dims):
  dims = validate_prefix_dims(prefix_dims)
  return (dims[0], *(b - a for a, b in zip(dims[:-1], dims[1:])))


def split_prefixes_blocks(z, prefix_dims):
  z = np.asarray(z)
  dims = validate_prefix_dims(prefix_dims, z.shape[-1])
  lows = (0, *dims[:-1])
  prefixes = tuple(z[..., :d] for d in dims)
  blocks = tuple(z[..., lo:hi] for lo, hi in zip(lows, dims))
  result = RepresentationSlices(z, prefixes, blocks, dims, block_dims(dims))
  assert np.array_equal(np.concatenate(result.blocks, -1), z)
  return result
