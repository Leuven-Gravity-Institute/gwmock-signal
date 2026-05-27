#
# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
"""Network optimal SNR for correlated detector networks."""

from __future__ import annotations

import numpy as np

from gwmock_signal.multichannel.stack import DetectorStrainStack


def network_optimal_snr(
    stack: DetectorStrainStack,
    psds: dict[str, np.ndarray],
    cross_psds: dict[tuple[str, str], np.ndarray] | None = None,
    low_frequency_cutoff: float = 20.0,
    high_frequency_cutoff: float | None = None,
) -> float:
    """Return the network optimal SNR for a correlated detector network.

    Computes ``sqrt((s|s))`` via Equations 9 and 18 of Cireddu et al. 2025
    (arXiv:2312.14614), where the frequency-domain inner product accounts for
    off-diagonal correlations in the spectral noise matrix.

    Args:
        stack: Multi-detector strain data as a ``DetectorStrainStack``.
        psds: One-sided noise PSD per detector; real numpy arrays of shape
            ``(n_freq,)`` where ``n_freq = n_samples // 2 + 1``.
            Values must be strictly positive (non-zero) at all bins;
            set sub-cutoff bins to `np.inf` to exclude them without causing singular matrix errors.
        cross_psds: Off-diagonal cross-PSDs, complex numpy arrays of shape
            ``(n_freq,)``, keyed by ordered detector-name tuples.  The
            function enforces Hermitian symmetry:
            ``s_n[l, m] = conj(s_n[m, l])``.  When ``None``, the noise matrix
            is block-diagonal (uncorrelated limit, Eq. 19).
        low_frequency_cutoff: Lower frequency bound in Hz (default 20 Hz).
        high_frequency_cutoff: Upper frequency bound in Hz; defaults to the
            Nyquist frequency when ``None``.

    Returns:
        Network optimal SNR as a non-negative float.
    """
    det_names = list(stack.detector_names)
    n_det = len(det_names)
    ref_ts = stack[det_names[0]]
    n_samples = len(ref_ts.value)
    dt = float(ref_ts.dt.value)
    delta_f = 1.0 / (n_samples * dt)
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    n_freq = len(freqs)

    # FFT each detector with dt normalisation (Whittle convention, Eq. 9)
    s_tilde = np.array([np.fft.rfft(stack[d].value) * dt for d in det_names])  # (n_det, n_freq)

    # Build spectral noise matrix s_n[i, j, k] for each frequency bin k
    s_n = np.zeros((n_det, n_det, n_freq), dtype=complex)
    for i, d_i in enumerate(det_names):
        s_n[i, i, :] = psds[d_i]
    if cross_psds:
        for (d_a, d_b), cpsd in cross_psds.items():
            i, j = det_names.index(d_a), det_names.index(d_b)
            s_n[i, j, :] = cpsd
            s_n[j, i, :] = np.conj(cpsd)  # Hermitian symmetry

    # Invert s_n at each frequency bin; np.linalg.inv broadcasts over the batch
    # axis. s_n is (n_det, n_det, n_freq); transpose to (n_freq, n_det, n_det)
    # for batched inversion, then transpose back with (1, 2, 0).
    s_n_inv = np.linalg.inv(s_n.transpose(2, 0, 1)).transpose(1, 2, 0)

    # Inner product integrand: Re[ s̃*(f_k) · s_n⁻¹(f_k) · s̃(f_k) ]  (Eq. 9)
    integrand = np.einsum("ik,ijk,jk->k", s_tilde.conj(), s_n_inv, s_tilde).real

    f_high = high_frequency_cutoff if high_frequency_cutoff is not None else freqs[-1]
    mask = (freqs >= low_frequency_cutoff) & (freqs <= f_high)

    rho_sq = 4.0 * delta_f * integrand[mask].sum()
    return float(np.sqrt(max(rho_sq, 0.0)))
