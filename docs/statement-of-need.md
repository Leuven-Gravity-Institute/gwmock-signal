# Statement of need

Next-generation observatories such as the Einstein Telescope are expected to
observe of order 10⁵ compact-binary coalescences per year, so mock data
challenges must scale from a handful of injections to full catalogues across a
detector network. Waveform physics is already well served by mature libraries
(`LALSuite`, `PyCBC`, and the JAX-based `ripple`), but each exposes a different
interface and none provides a single, batched assembly path from source
parameters to multi-detector strain. `gwmock-signal` fills that gap: it wraps
these generators behind one stable backend boundary
(`GWSimulator.simulate(...) -> DetectorStrainStack`), so pipelines can switch
waveform backends without code changes, and it adds a GPU-batched path that
assembles many events onto a shared frequency grid for catalogue-scale runs. It
is the signal layer of the `gwmock` mock-data-challenge ecosystem and can be
used either standalone or through the `gwmock` orchestrator.
