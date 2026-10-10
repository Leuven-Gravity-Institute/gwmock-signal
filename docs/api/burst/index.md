---
title: Burst
description: Burst simulator, ad hoc burst waveforms and the test injection set.
icon: material/flash
---

<!-- prettier-ignore-start -->

::: gwmock_signal.burst
    options:
        show_root_heading: true
        heading_level: 2
        inherited_members: true
        show_if_no_docstring: false
        docstring_style: google
        show_source: true

::: gwmock_signal.burst.waveforms
    options:
        show_root_heading: true
        heading_level: 2
        show_if_no_docstring: false
        docstring_style: google
        show_source: true

<!-- prettier-ignore-end -->

`BurstSimulator` is a [`TransientSimulator`](../simulator/): each burst is
generated, projected onto the network and injected like any other transient. It
is registered under the `source_type` `"burst"`.

For the injection specification and usage examples, see the
[User guide — Burst injections](../../user_guide/burst-injections.md).
