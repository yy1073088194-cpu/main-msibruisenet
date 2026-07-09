"""Pluggable feature enhancement modules.

Modules:
    - SEBlock: Channel attention via global average pooling (Squeeze-and-Excitation).
    - SpectralConv1D: 1D convolution along the spectral (band) dimension.
    - DiagonalBandGate: Static hard top-k band selection at the network input.
    - ConcreteSelect: Concrete-autoencoder-style differentiable band selection
      (baseline; same interface as DiagonalBandGate, concrete/Gumbel relaxation).
"""

import math

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Performs: GAP -> FC(C, C//r) -> ReLU -> FC(C//r, C) -> Sigmoid -> channel-wise scaling.

    Args:
        channels: Number of input/output channels.
        reduction: Reduction ratio for the bottleneck (default 16).
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class SpectralConv1D(nn.Module):
    """1D convolution along the spectral (band) dimension.

    Learns local correlations between adjacent NIR bands (23nm spacing).
    Includes a residual connection.

    Args:
        num_channels: Number of channels to process.
        kernel_size: 1D convolution kernel size (default 3).
    """

    def __init__(self, num_channels=16, kernel_size=3):
        super().__init__()
        self.num_channels = num_channels
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=True)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        x_flat = x_flat.reshape(B * H * W, 1, C)
        x_flat = self.conv(x_flat)
        x_flat = x_flat.reshape(B, H * W, C).permute(0, 2, 1)
        x_out = x_flat.view(B, C, H, W)
        return self.relu(self.bn(x_out + x))


class DiagonalBandGate(nn.Module):
    """Prior-free diagonal band-selection gate with straight-through hard top-k.

    Selects exactly ``k`` of ``num_bands`` input channels. The gate is a single
    learnable score vector ``theta`` (one scalar per band), so the same bands are
    kept for every image — a *selection* mechanism (read off the kept bands and
    physically drop the others at deployment), not the per-sample reweighting of
    an SE block. No spectral prior is injected; the selection is purely data
    driven from the segmentation loss.

    Score / soft-prob / mask::

        s_b   = theta_b
        g_b   = sigmoid(s_b / tau)              # soft keep-prob (gradient path)
        m     = top-k(s)                        # hard k-hot mask
        m_st  = m + g - g.detach()              # straight-through estimator

    During training the forward pass applies the HARD mask ``m`` (the backbone
    always sees exactly k bands, matching deployment), while gradients flow to
    ALL bands through ``g`` — so an unselected band can still earn its way in,
    instead of the gate locking onto its initialization. ``tau`` is annealed by
    :meth:`set_progress` (call once per epoch with ``frac`` in [0, 1]) from
    ``tau_start`` to ``tau_end`` (soft/explore -> near-hard/commit).

    With ``k == num_bands`` the gate is the identity (all bands pass), so
    ``M_{k=B}`` is exactly the full-input baseline.

    Args:
        num_bands: Number of input channels (= encoder ``in_channels``).
        k: Number of bands to keep (1 <= k <= num_bands).
        tau_start, tau_end: Temperature annealing endpoints.
        theta_init_noise: Std of Gaussian theta init (breaks top-k ties).
        random_select: If True, freeze the gate on a fixed *random* k-subset
            (untrained) — a same-architecture control showing the learned
            selection is not arbitrary.
    """

    def __init__(self, num_bands, k,
                 tau_start=1.0, tau_end=0.05,
                 theta_init_noise=0.01, random_select=False):
        super().__init__()
        if not (1 <= k <= num_bands):
            raise ValueError(f"k must be in [1, {num_bands}], got {k}")

        self.num_bands = num_bands
        self.k = k
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.random_select = bool(random_select)

        if theta_init_noise > 0:
            theta0 = theta_init_noise * torch.randn(num_bands)
        else:
            theta0 = torch.zeros(num_bands)
        self.theta = nn.Parameter(theta0)

        if self.random_select:
            # Fixed random k-subset; frozen (no learning) -> "not arbitrary" control.
            with torch.no_grad():
                self.theta.zero_()
                idx = torch.randperm(num_bands)[:k]
                self.theta[idx] = 1.0
            self.theta.requires_grad_(False)

        # Schedule state; defaults to fully-annealed so a freshly loaded model
        # (e.g. in eval.py without any set_progress call) behaves as deployment.
        self.register_buffer("_tau", torch.tensor(self.tau_end))

    def set_progress(self, frac):
        """Update tau for the current training progress (frac in [0, 1])."""
        frac = float(min(max(frac, 0.0), 1.0))
        cos = 0.5 * (1.0 + math.cos(math.pi * frac))  # 1 -> 0 as frac 0 -> 1
        self._tau.fill_(self.tau_end + (self.tau_start - self.tau_end) * cos)

    def _hard_mask(self, scores):
        idx = torch.topk(scores, self.k).indices
        mask = torch.zeros_like(scores)
        mask[idx] = 1.0
        return mask

    def forward(self, x):
        scores = self.theta
        if self.training and not self.random_select:
            tau = self._tau.clamp_min(1e-4)
            g = torch.sigmoid(scores / tau)
            mask = self._hard_mask(scores)
            mask = mask + g - g.detach()  # straight-through: forward=hard, grad=soft
        else:
            mask = self._hard_mask(scores)
        return x * mask.view(1, -1, 1, 1)

    @torch.no_grad()
    def selected_bands(self):
        """Return the sorted list of kept band indices."""
        return torch.topk(self.theta, self.k).indices.sort().values.tolist()

    def extra_repr(self):
        mode = "random" if self.random_select else "learned"
        return (f"num_bands={self.num_bands}, k={self.k}, mode={mode}, "
                f"tau={self.tau_start}->{self.tau_end}")


class ConcreteSelect(nn.Module):
    """Concrete-autoencoder-style differentiable band selection (Balin et al. 2019).

    A drop-in sibling of :class:`DiagonalBandGate`: it also selects exactly ``k``
    of ``num_bands`` input channels and exposes the same ``forward`` /
    ``set_progress`` / ``selected_bands`` API, so it slots into the existing
    ``band_gate`` position and inherits the per-epoch tau schedule, the
    selected-bands logging (``selected_bands.json``) and the deploy export with
    no other plumbing changes.

    The ONLY thing that differs from the hard top-k + STE gate is the *relaxation*
    of the discrete choice, which is exactly the variable this baseline isolates:

        * DiagonalBandGate : one shared score vector, hard top-k, straight-through
          (forward = hard k-hot).
        * ConcreteSelect   : ``k`` independent Concrete / Gumbel-softmax selector
          nodes; forward = soft convex weights that anneal to one-hot.

    Node ``j`` holds logits ``alpha[j]`` over the ``B`` bands. In training::

        w_j = softmax((alpha_j + gumbel) / tau)     # Concrete sample, sum_b w_jb = 1
        m   = max_j w_j                             # per-band keep-mask (B,)
        x'  = x * m                                 # B channels, ~k active

    so gradients flow to every band and the mask sharpens to k-hot as ``tau`` -> 0.
    At inference each node takes ``argmax(alpha_j)``; the distinct picks form the
    hard k-hot mask, and any collisions are filled from the next-highest logits so
    exactly ``k`` bands are always kept (identical output form to the gate, hence
    the same exact-pruning deploy path).

    Args:
        num_bands: Number of input channels (= encoder ``in_channels``).
        k: Number of bands to keep (1 <= k <= num_bands).
        tau_start, tau_end: Temperature annealing endpoints (match the gate).
        alpha_init_noise: Std of Gaussian logit init (breaks argmax ties).
    """

    def __init__(self, num_bands, k,
                 tau_start=1.0, tau_end=0.05, alpha_init_noise=0.01):
        super().__init__()
        if not (1 <= k <= num_bands):
            raise ValueError(f"k must be in [1, {num_bands}], got {k}")

        self.num_bands = num_bands
        self.k = k
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)

        self.alpha = nn.Parameter(alpha_init_noise * torch.randn(k, num_bands))

        # Fully-annealed default so a freshly loaded model (e.g. eval.py without
        # any set_progress call) behaves as deployment (hard selection).
        self.register_buffer("_tau", torch.tensor(self.tau_end))

    def set_progress(self, frac):
        """Update tau for the current training progress (frac in [0, 1])."""
        frac = float(min(max(frac, 0.0), 1.0))
        cos = 0.5 * (1.0 + math.cos(math.pi * frac))  # 1 -> 0 as frac 0 -> 1
        self._tau.fill_(self.tau_end + (self.tau_start - self.tau_end) * cos)

    def _hard_indices(self):
        """Distinct argmax per node, filled to exactly k from the global ranking."""
        picks = torch.argmax(self.alpha, dim=1).tolist()
        seen = []
        for b in picks:
            if b not in seen:
                seen.append(int(b))
        if len(seen) < self.k:
            order = torch.argsort(self.alpha.max(dim=0).values,
                                  descending=True).tolist()
            for b in order:
                if b not in seen:
                    seen.append(int(b))
                if len(seen) == self.k:
                    break
        return sorted(seen[:self.k])

    def forward(self, x):
        if self.training:
            tau = self._tau.clamp_min(1e-4)
            u = torch.rand_like(self.alpha).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            w = torch.softmax((self.alpha + gumbel) / tau, dim=1)  # (k, B)
            mask = w.max(dim=0).values                             # (B,)
        else:
            idx = self._hard_indices()
            mask = torch.zeros(self.num_bands, device=x.device, dtype=x.dtype)
            mask[idx] = 1.0
        return x * mask.view(1, -1, 1, 1)

    @torch.no_grad()
    def selected_bands(self):
        """Return the sorted list of kept band indices."""
        return self._hard_indices()

    def extra_repr(self):
        return (f"num_bands={self.num_bands}, k={self.k}, "
                f"tau={self.tau_start}->{self.tau_end}, mode=concrete")
