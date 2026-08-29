import numpy as np


def flatten_rssm_state(state):
  deter, stoch = np.asarray(state['deter']), np.asarray(state['stoch'])
  return np.concatenate([deter, stoch.reshape((*stoch.shape[:-2], -1))], -1)


def unflatten_rssm_state(h, deter_dim, stoch_shape):
  h = np.asarray(h)
  deter_dim, stoch_shape = int(deter_dim), tuple(map(int, stoch_shape))
  expected = deter_dim + int(np.prod(stoch_shape))
  if h.shape[-1] != expected:
    raise ValueError(f'Expected h dim {expected}, got {h.shape[-1]}')
  return {
      'deter': h[..., :deter_dim],
      'stoch': h[..., deter_dim:].reshape((*h.shape[:-1], *stoch_shape)),
  }


def normalize_ground_truth(image):
  image = np.asarray(image)
  if image.dtype != np.uint8:
    raise TypeError(f'Expected uint8 image, got {image.dtype}')
  return image.astype(np.float32) / 255.0


def assert_image_domain(image, name='image'):
  image = np.asarray(image)
  if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
    raise ValueError(f'{name} is outside [0, 1]')
