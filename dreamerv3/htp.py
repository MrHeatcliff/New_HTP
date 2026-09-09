"""
Hierarchical Temporal Prefix World Model (HTP-WM).

This module implements HTP-WM as an *auxiliary* representation-learning
objective added to a base world model (DreamerV3's RSSM). It reorganises the
backbone latent h_t into an ordered representation

    z_t = S_ψ(h_t)  in  R^D,

with cumulative prefixes z_t^(1:ℓ) = z_t[:d_ℓ] for 0 = d_0 < d_1 < ... < d_L = D.

Two auxiliary losses induce the temporal-hierarchical structure:

  (1) Progressive prefix reconstruction (Eq. 4 in the paper).
      Learned residual heads G_ℓ map each newly introduced block z^(ℓ) to a
      contribution Δĥ^(ℓ) toward reconstructing sg[h_t]. A stop-gradient on the
      cumulative reconstruction prevents higher-level losses from directly
      updating earlier residual contributions.

  (2) Multi-stride prefix dynamics (Eq. 7 in the paper).
      A per-level predictor F_ω^(ℓ) predicts the *future* prefix
      z̄_{t+Δ_ℓ}^(1:d_ℓ) from the current online prefix z̃_t^(1:d_ℓ) and the
      intervening action sequence a_{t:t+Δ_ℓ-1}. Compact prefixes are assigned
      *longer* strides Δ_ℓ; larger prefixes handle *shorter* strides.

Behaviour learning still uses the full ordered representation z_t^(1:L) as
input to actor/critic/reward/continuation heads. Latent imagination continues
to rely on the backbone's one-step dynamics (self.dyn.imagine in agent.py);
the multi-stride predictors are auxiliary only and are not used to unroll
imagined trajectories.
"""

import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
sg = jax.lax.stop_gradient


def causal_episode_average(x, reset, window):
    """Past-only feature average, truncated at the latest episode reset."""
    if window < 1:
        raise ValueError('Averaging window must be positive')
    if window == 1 or x.shape[1] == 0:
        return x
    if reset is None or reset.shape != x.shape[:2]:
        raise ValueError('Temporal reconstruction target requires matching reset flags')
    time = jnp.arange(x.shape[1])[None, :]
    last_reset = jax.lax.associative_scan(
        jnp.maximum, jnp.where(reset, time, 0), axis=1)
    start = jnp.maximum(last_reset, time - window + 1)
    sums = jnp.concatenate([
        jnp.zeros_like(x[:, :1], dtype=f32), jnp.cumsum(x.astype(f32), axis=1)], axis=1)
    previous = jnp.take_along_axis(sums, start[..., None], axis=1)
    return ((sums[:, 1:] - previous) / (time - start + 1)[..., None]).astype(x.dtype)


def prefix_isotropy(z, dim):
    """Scale-free covariance shape penalty; discourages dimensional collapse."""
    x = z[..., :dim].astype(f32).reshape((-1, dim))
    x = x - x.mean(0)
    covariance = x.T @ x / max(x.shape[0], 1)
    normalized = covariance / jnp.maximum(jnp.trace(covariance) / dim, 1e-8)
    return jnp.square(normalized - jnp.eye(dim)).sum() / dim


def prefix_participation_rank(z, dim):
    """Covariance participation rank, including a zero for exact constants."""
    x = z[..., :dim].astype(f32).reshape((-1, dim))
    x = x - x.mean(0)
    covariance = x.T @ x / max(len(x), 1)
    covariance /= jnp.maximum(jnp.trace(covariance) / dim, 1e-8)
    return jnp.trace(covariance) ** 2 / jnp.maximum(jnp.square(covariance).sum(), 1e-12)


def prefix_rank_floor(z, dim, min_rank):
    """Stop pushing covariance toward isotropy once participation rank suffices.

    For nondegenerate covariance below the floor, gradients match isotropy:
    isotropy = dim / participation_rank - 1. Constants have finite positive
    loss but, as with covariance penalties generally, zero escape gradient.
    """
    if not 1 <= min_rank <= dim:
        raise ValueError('min_rank must lie between 1 and prefix dimension')
    rank = prefix_participation_rank(z, dim)
    return jnp.maximum(dim / jnp.maximum(rank, 1.0) - dim / min_rank, 0.0)


def prefix_block_redundancy(z, first_dim, second_end):
    """Linear CKA between the first two blocks; only block 2 gets direct gradients.

    Scale invariant away from numerical floors. Zero-valued blocks are a
    degenerate minimum, so reconstruction and rank diagnostics remain needed.
    Shared projection parameters can still indirectly change block 1.
    """
    if not 0 < first_dim < second_end <= z.shape[-1]:
        raise ValueError('Require two nonempty blocks within the representation')
    a = sg(z[..., :first_dim].astype(f32).reshape((-1, first_dim)))
    b = z[..., first_dim:second_end].astype(f32).reshape((-1, second_end - first_dim))
    a, b = a - a.mean(0), b - b.mean(0)
    # Normalize RMS first to keep the covariance products well conditioned.
    a /= jnp.sqrt(jnp.maximum(jnp.square(a).mean(), 1e-8))
    b /= jnp.sqrt(jnp.maximum(jnp.square(b).mean(), 1e-8))
    n = max(len(a), 1)
    cross = a.T @ b / n
    ca, cb = a.T @ a / n, b.T @ b / n
    denominator = jnp.sqrt(jnp.maximum(jnp.square(ca).sum() * jnp.square(cb).sum(), 1e-12))
    return jnp.square(cross).sum() / denominator


def prefix_persistence(z, reset, dim, stride, all_lags=False, metric='pooled'):
    """Scale-normalized temporal variation, excluding episode boundaries.

    This is a soft slowness bias, not a semantic-invariance guarantee.
    Use alongside information-preserving objectives; constants remain degenerate.
    """
    if metric not in ('pooled', 'whitened'):
        raise ValueError(f'Unknown persistence metric: {metric}')
    if all_lags:
        return jnp.stack([
            prefix_persistence(z, reset, dim, lag, metric=metric)
            for lag in range(1, min(stride, z.shape[1] - 1) + 1)
        ]).mean() if z.shape[1] > 1 else jnp.asarray(0.0, f32)
    x = z[..., :dim].astype(f32)
    if x.shape[1] <= stride:
        return jnp.asarray(0.0, f32)
    episode = jnp.cumsum(reset.astype(jnp.int32), axis=1)
    valid = (episode[:, :-stride] == episode[:, stride:]).astype(f32)
    count = valid.sum()
    a, b = x[:, :-stride], x[:, stride:]
    mean = ((a + b) * valid[..., None]).sum((0, 1)) / jnp.maximum(2 * count, 1)
    if metric == 'whitened':
        # Approximate Mahalanobis variation. Unlike a ratio of traces, this
        # does not reward merely concentrating variance in the slowest axis.
        # Relative ridge stabilizes solves; extreme anisotropy and exact
        # collapse still require separate diagnostics/anti-collapse losses.
        ac = ((a - mean) * valid[..., None]).reshape((-1, dim))
        bc = ((b - mean) * valid[..., None]).reshape((-1, dim))
        delta = ((a - b) * valid[..., None]).reshape((-1, dim))
        covariance = (ac.T @ ac + bc.T @ bc) / jnp.maximum(2 * count, 1)
        difference = delta.T @ delta / jnp.maximum(count, 1)
        ridge = jnp.maximum(jnp.trace(covariance) / dim * 1e-4, 1e-8)
        return jnp.trace(jnp.linalg.solve(
            covariance + ridge * jnp.eye(dim), difference)) / (2 * dim)
    variance = (((a - mean) ** 2 + (b - mean) ** 2) * valid[..., None]).sum()
    variation = (((a - b) ** 2) * valid[..., None]).sum()
    return variation / jnp.maximum(variance, 1e-8)


def cumulative_residual_reconstruction(residuals, template):
    """Apply the exact training recurrence h_l = sg(h_{l-1}) + G_l."""
    current = jnp.zeros_like(template)
    outputs = []
    for residual in residuals:
        current = sg(current) + residual
        outputs.append(current)
    return tuple(outputs)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class OrderedProjection(nj.Module):
    """
    Learned projection S_ψ : R^{feat_dim} -> R^D whose output coordinates are
    partitioned into contiguous blocks z_t = [z^(1), z^(2), ..., z^(L)] such
    that the cumulative prefix z^(1:ℓ) has dimension d_ℓ.

    Two variants:
      * Shared trunk with a single output layer (default). Provides tight
        parameter sharing across blocks. Prefix nesting is *activation-level*.
      * Block-separable: each output block has its own linear head applied on
        top of a shared trunk. This yields stronger *parameter-level*
        isolation across blocks (Eq. 37 in the paper).
    """

    dims: tuple = (128, 256, 512, 1024, 2048)   # cumulative prefix boundaries d_1..d_L
    hidden: int = 512
    layers: int = 2
    norm: str = 'rms'
    act: str = 'gelu'
    separable: bool = False
    outscale: float = 1.0

    def __init__(self, **kw):
        assert all(a < b for a, b in zip(self.dims[:-1], self.dims[1:])), (
            'dims must be strictly increasing (cumulative prefixes): %s' % (self.dims,))
        self.kw = kw

    @property
    def output_dim(self):
        return self.dims[-1]

    @property
    def block_sizes(self):
        """Sizes of the *newly introduced* blocks: [d_1, d_2 - d_1, ..., d_L - d_{L-1}]."""
        return tuple(
            [self.dims[0]] +
            [b - a for a, b in zip(self.dims[:-1], self.dims[1:])])

    @property
    def num_levels(self):
        return len(self.dims)

    def __call__(self, h):
        """h : (..., feat_dim) -> z : (..., D)."""
        x = h
        for i in range(self.layers):
            x = self.sub(f'trunk{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(
                self.sub(f'trunk{i}norm', nn.Norm, self.norm)(x))

        if self.separable:
            outs = []
            for i, block_size in enumerate(self.block_sizes):
                head_kw = dict(**self.kw, outscale=self.outscale)
                outs.append(self.sub(
                    f'block{i}', nn.Linear, block_size, **head_kw)(x))
            z = jnp.concatenate(outs, -1)
        else:
            head_kw = dict(**self.kw, outscale=self.outscale)
            z = self.sub('out', nn.Linear, self.output_dim, **head_kw)(x)
        return z


class ResidualReconHead(nj.Module):
    """
    Residual reconstruction head G_ℓ : R^{block_ℓ} -> R^{feat_dim}.

    Produces the level-ℓ increment Δĥ^(ℓ) added to the running reconstruction
    ĥ^(ℓ) = sg[ĥ^(ℓ-1)] + Δĥ^(ℓ). The stop-gradient is applied *outside*
    (in ProgressiveRecon) so that this head has no notion of it.
    """

    hidden: int = 512
    layers: int = 1
    norm: str = 'rms'
    act: str = 'gelu'
    outscale: float = 1.0

    def __init__(self, out_dim, **kw):
        self.out_dim = out_dim
        self.kw = kw

    def __call__(self, block):
        x = block
        for i in range(self.layers):
            x = self.sub(f'hid{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(
                self.sub(f'hid{i}norm', nn.Norm, self.norm)(x))
        head_kw = dict(**self.kw, outscale=self.outscale)
        delta = self.sub('out', nn.Linear, self.out_dim, **head_kw)(x)
        return delta


class PrefixDynamics(nj.Module):
    """
    Stride-specific prefix dynamics predictor F_ω^(ℓ).

    Takes the current prefix z̃_t^(1:d_ℓ) and a *flat* action-sequence tensor of
    shape (..., stride * act_flat_dim), and predicts z̄_{t+Δ_ℓ}^(1:d_ℓ).

    We use a plain MLP over the concatenated action sequence. For Atari with
    ~18 discrete actions and stride 16 this is a 288-D input concatenated to
    a 128-D prefix -> 416-D input to the MLP, comfortably small.
    """

    hidden: int = 512
    layers: int = 2
    norm: str = 'rms'
    act: str = 'gelu'
    outscale: float = 1.0

    def __init__(self, prefix_dim, **kw):
        self.prefix_dim = prefix_dim
        self.kw = kw

    def __call__(self, prefix, action_seq):
        """
        Args:
          prefix     : (..., prefix_dim)                 online prefix z̃_t^(1:d_ℓ)
          action_seq : (..., stride, act_flat_dim)       intervening actions
        Returns:
          pred       : (..., prefix_dim)                 predicted future prefix
        """
        actions_flat = action_seq.reshape((*action_seq.shape[:-2], -1))
        x = jnp.concatenate([prefix, actions_flat], -1)
        for i in range(self.layers):
            x = self.sub(f'hid{i}', nn.Linear, self.hidden, **self.kw)(x)
            x = nn.act(self.act)(
                self.sub(f'hid{i}norm', nn.Norm, self.norm)(x))
        head_kw = dict(**self.kw, outscale=self.outscale)
        return self.sub('out', nn.Linear, self.prefix_dim, **head_kw)(x)


# ---------------------------------------------------------------------------
# Containers for L residual heads / L stride-specific predictors
# ---------------------------------------------------------------------------


class ProgressiveRecon(nj.Module):
    """
    Container for L residual reconstruction heads G_1, ..., G_L implementing
    Eq. 4 of the paper.

    Given z_t (online) and h_t (base representation; stop-gradient applied by
    the caller), returns:
      loss       : (B, T)      weighted sum β_ℓ * ||ĥ^(ℓ) - h||^2 / d_h
      per_level  : list[(B, T)]  raw level-wise reconstruction errors
    """

    dims: tuple = (128, 256, 512, 1024, 2048)
    hidden: int = 512
    layers: int = 1
    norm: str = 'rms'
    act: str = 'gelu'
    outscale: float = 1.0

    first_target_window: int = 1

    def __init__(self, feat_dim, **kw):
        if self.first_target_window < 1:
            raise ValueError('first_target_window must be positive')
        self.feat_dim = feat_dim
        self.kw = kw

    @property
    def num_levels(self):
        return len(self.dims)

    def _block(self, z, ell):
        lo = 0 if ell == 0 else self.dims[ell - 1]
        hi = self.dims[ell]
        return z[..., lo:hi]

    def reconstruct(self, z, h_template):
        """Return cumulative reconstructions without computing a loss.

        This is the read-only evaluation path. Numerically it is identical to
        the training reconstruction recurrence, including the stop-gradient
        between cumulative levels, but it does not consume a target.
        """
        residuals = []
        for ell in range(self.num_levels):
            block = self._block(z, ell)
            head = self.sub(
                f'head{ell}', ResidualReconHead,
                out_dim=self.feat_dim, hidden=self.hidden,
                layers=self.layers, norm=self.norm, act=self.act,
                outscale=self.outscale, **self.kw)
            residuals.append(head(block))
        return cumulative_residual_reconstruction(residuals, h_template)

    def __call__(self, z, h_target, reset=None):
        """
        z        : (B, T, D)          online ordered representation
        h_target : (B, T, feat_dim)   base representation (sg applied by caller)
        """
        L = self.num_levels
        beta = 1.0 / L  # uniform level weights
        cumulative = self.reconstruct(z, h_target)
        coarse_target = causal_episode_average(h_target, reset, self.first_target_window)
        per_level = []
        total = jnp.zeros(h_target.shape[:-1], dtype=h_target.dtype)
        for ell, h_recon in enumerate(cumulative):
            target = coarse_target if ell == 0 else h_target
            err = ((target - h_recon) ** 2).mean(-1)  # (B, T); ||·||^2 / d_h
            per_level.append(err)
            total = total + beta * err
        return total, per_level


class MultiStridePDyn(nj.Module):
    """
    Container for L stride-specific prefix dynamics predictors, implementing
    Eq. 7 of the paper.

    For each level ℓ we:
      * construct the *gradient-isolated* input prefix
            z̃^(1:d_ℓ)_t = [sg(z^(1:d_{ℓ-1})_t), z^(d_{ℓ-1}:d_ℓ)_t]
        so that only the *newly introduced* block receives gradients from the
        level-ℓ objective while previous blocks contribute as frozen context;
      * gather sliding action windows a_{t:t+Δ_ℓ-1};
      * predict the future prefix and match it to the slow-target prefix at
        time t + Δ_ℓ.

    The loss for the last Δ_ℓ time steps of each batch is undefined (no target
    available) and is masked out by padding those positions with zero. This
    keeps the returned tensor shape (B, T) so it can be added to other losses
    in agent.py.
    """

    dims: tuple = (128, 256, 512, 1024, 2048)
    strides: tuple = (16, 8, 4, 2, 1)
    hidden: int = 512
    layers: int = 2
    norm: str = 'rms'
    act: str = 'gelu'
    outscale: float = 1.0
    enforce_coarse_to_fine: bool = True
    mask_episode_boundaries: bool = False  # compatibility default; explicit ablation

    def __init__(self, **kw):
        assert len(self.dims) == len(self.strides), (
            'dims and strides must have equal length: %s vs %s'
            % (self.dims, self.strides))
        assert all(s >= 1 for s in self.strides), self.strides
        # Compact prefixes should be assigned longer strides.
        if self.enforce_coarse_to_fine:
            assert all(a >= b for a, b in zip(self.strides[:-1], self.strides[1:])), (
                'strides must be non-increasing (compact prefix -> longer stride): %s'
                % (self.strides,))
        # NOTE: act_flat_dim is inferred lazily by nn.Linear at first call,
        # so we don't need to know it here.
        self.kw = kw

    @property
    def num_levels(self):
        return len(self.dims)

    def _gi_prefix(self, z, ell):
        """z̃^(1:d_ℓ)_t with activation-level gradient isolation (Eq. 5/29)."""
        if ell == 0:
            return z[..., :self.dims[0]]
        prev = sg(z[..., :self.dims[ell - 1]])
        curr = z[..., self.dims[ell - 1]:self.dims[ell]]
        return jnp.concatenate([prev, curr], -1)

    @staticmethod
    def temporal_indices(length, stride):
        """Indices used by the real prediction-loss path for one level."""
        valid = int(length) - int(stride)
        if valid <= 0:
            return (
                jnp.zeros((0,), jnp.int32),
                jnp.zeros((0, int(stride)), jnp.int32),
                jnp.zeros((0,), jnp.int32),
            )
        source = jnp.arange(valid, dtype=jnp.int32)
        offsets = jnp.arange(int(stride), dtype=jnp.int32)
        actions = source[:, None] + offsets[None, :]
        targets = source + int(stride)
        return source, actions, targets

    def __call__(self, z_online, z_target_slow, action_seq_full, reset=None):
        """
        Args:
          z_online         : (B, T, D)                  online ordered representation
          z_target_slow    : (B, T, D)                  slow-target ordered representation
          action_seq_full  : (B, T, act_flat_dim)       flat action embeddings, where
              index t corresponds to a_t (the action taken FROM state t
              leading to state t+1).
        Returns:
          loss       : (B, T)     α-weighted sum over levels of the per-level
                                  prediction error, zero-padded on the tail
          per_level  : list[dict] with keys 'stride', 'err_mean'
        """
        L = self.num_levels
        alpha = 1.0 / L
        B, T, _ = z_online.shape
        if self.mask_episode_boundaries and reset is None:
            raise ValueError('Episode-masked dynamics requires reset flags')
        if reset is not None and reset.shape != (B, T):
            raise ValueError(f'Expected reset shape {(B, T)}, got {reset.shape}')
        episode = (jnp.cumsum(reset.astype(jnp.int32), axis=1)
                   if reset is not None else None)
        total = jnp.zeros((B, T), dtype=z_online.dtype)
        per_level = []
        for ell in range(L):
            Delta = int(self.strides[ell])
            d_ell = int(self.dims[ell])
            T_valid = T - Delta
            if T_valid <= 0:
                per_level.append({'stride': Delta, 'err_mean': jnp.array(0.0)})
                continue

            # Input prefix at time t (activation-level gradient isolation).
            prefix_all = self._gi_prefix(z_online, ell)  # (B, T, d_ell)
            prefix_in = prefix_all[:, :T_valid]          # (B, T_valid, d_ell)

            # Sliding action windows a_{t:t+Δ-1}.
            t_idx, gather, target_idx = self.temporal_indices(T, Delta)
            actions_windowed = action_seq_full[:, gather]                # (B, T_valid, Δ, act_flat_dim)

            predictor = self.sub(
                f'head{ell}', PrefixDynamics,
                prefix_dim=d_ell, hidden=self.hidden, layers=self.layers,
                norm=self.norm, act=self.act, outscale=self.outscale, **self.kw)
            pred = predictor(prefix_in, actions_windowed)                 # (B, T_valid, d_ell)

            # Slow-target prefix at time t + Δ (sg applied to be safe).
            target = sg(z_target_slow[:, target_idx, :d_ell])             # (B, T_valid, d_ell)

            err = ((pred - target) ** 2).mean(-1)                          # (B, T_valid)
            valid = (episode[:, :T_valid] == episode[:, Delta:]
                     if episode is not None else jnp.ones((B, T_valid), bool))
            if self.mask_episode_boundaries:
                err = jnp.where(valid, err, 0)
                err_mean = err.astype(f32).sum() / jnp.maximum(valid.sum(), 1)
            else:
                err_mean = err.mean()
            per_level.append({'stride': Delta, 'err_mean': err_mean,
                              'cross_episode_fraction': 1 - valid.astype(f32).mean()})

            # Zero-pad the last Δ positions so the returned loss is (B, T).
            pad = jnp.zeros((B, Delta), dtype=err.dtype)
            level_loss = jnp.concatenate([err, pad], axis=1)               # (B, T)
            total = total + alpha * level_loss
        return total, per_level


# ---------------------------------------------------------------------------
# Helper: flatten an action-dict into (B, T, act_flat_dim)
# ---------------------------------------------------------------------------


def flatten_action_dict(action_dict, act_space):
    """
    Convert a dict of per-timestep actions into a flat float tensor of shape
    (B, T, act_flat_dim). Uses nn.DictConcat on a reshaped batch so it works
    on arbitrary (B, T, ...) leading shapes.

    Args:
      action_dict : {name: (B, T, ...) tensor}. For discrete actions the
                    trailing dims are the one-hot representation (as produced
                    by DreamerV3's replay / policy sampling).
      act_space   : {name: elements.Space}
    """
    keys = list(action_dict.keys())
    first = action_dict[keys[0]]
    B, T = first.shape[:2]
    flat_in = {k: v.reshape((B * T, *v.shape[2:])) for k, v in action_dict.items()}
    flat_out = nn.DictConcat(act_space, 1)(flat_in)  # (B*T, act_flat_dim)
    return flat_out.reshape((B, T, -1))


# -- Note --
# The flat action-embedding dimensionality does not need to be inferred
# statically because ninjax's nn.Linear layers are input-lazy: they build
# their kernel on first call using the actual input shape. Just pass an
# action tensor through PrefixDynamics and it works.
