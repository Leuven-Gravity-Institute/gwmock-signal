---
title: Continuous waves
description: Continuous-wave (isolated pulsar) API.
icon: material/egg
---

<!-- prettier-ignore-start -->

::: gwmock_signal.continuous
    options:
        show_root_heading: true
        heading_level: 2
        inherited_members: true
        show_if_no_docstring: false
        docstring_style: google
        show_source: true

<!-- prettier-ignore-end -->

`ContinuousWaveSimulator` is a direct [`GWSimulator`](../simulator/) subclass,
not a `TransientSimulator` as a continuous wave is not a transient
placed at a coalescence time but a signal that is *on* for the whole observation, which is
typically months. Therefore, the simulation does not fit the transient injection pipeline:
*waveforms → projection → injection* used for CBC
signals. It is generated as **one analysis segment at a time**, against a **fixed SSB reference epoch**, so
consecutive segments join up into a single phase-coherent signal rather than
independent ones.

It requires the **JAX optional dependency** (`pip install 'gwmock-signal[jax]'`) as its polarizations come from
**rippleGW** (and JAX)'s pulsar-signal generator.

For **usage examples**, see the
[User guide — Continuous waves examples](../../user_guide/continuous-waves.md).
