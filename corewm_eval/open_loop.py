import numpy as np

from .metrics import assert_nonnegative


def validate_shared_backbone_rollout(h_tilde, prefix_reconstructions):
  h_tilde = np.asarray(h_tilde)
  for recon in prefix_reconstructions:
    if np.asarray(recon).shape != h_tilde.shape:
      raise ValueError((h_tilde.shape, np.asarray(recon).shape))
  return True


def rollout_error_families(gt, full, prefix_images, h_tilde, prefix_h):
  gt, full, h_tilde = map(lambda x: np.asarray(x, np.float64), (gt, full, h_tilde))
  e_backbone = np.abs(gt - full).mean(axis=tuple(range(1, gt.ndim)))
  rows = []
  for image, hhat in zip(prefix_images, prefix_h):
    image, hhat = np.asarray(image, np.float64), np.asarray(hhat, np.float64)
    row = {
        'E_h': np.square(h_tilde - hhat).mean(axis=-1),
        'E_prefix': np.abs(full - image).mean(axis=tuple(range(1, full.ndim))),
        'E_backbone': e_backbone,
        'E_total': np.abs(gt - image).mean(axis=tuple(range(1, gt.ndim))),
    }
    for name, values in row.items():
      assert_nonnegative(name, values)
    rows.append(row)
  return rows
