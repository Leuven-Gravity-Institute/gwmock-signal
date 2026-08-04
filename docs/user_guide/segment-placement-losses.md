# How much signal a segment boundary costs

A compact binary's buffer begins seconds before its coalescence. Data written in
segments therefore has a choice to make: if the segment that claims an event is
the one containing `coa_time`, then a buffer starting before that segment is
cropped, and the earlier segments it belonged to have already been written.

[`WaveformBackend.pre_coalescence_duration`][gwmock_signal.waveform.WaveformBackend.pre_coalescence_duration]
exists so a caller can claim the event by where its waveform _starts_ instead.
This page records what the cropping costs when it does not.

## The loss is not a single number

Every figure below is a fraction of the unweighted strain-squared energy, and
every one of them moves by **orders of magnitude** with three things:

- the low-frequency cutoff,
- how far past the boundary `coa_time` lands,
- the waveform backend.

A percentage quoted without all three is not interpretable. The tables therefore
name them, and so should anything that cites the tables.

## A 30+25 solar-mass binary on LAL

IMRPhenomD, 1024 Hz, into a single detector (H1), `coa_time` 0.5 s past a
segment boundary:

|              | 20 Hz cutoff    | 30 Hz cutoff    |
| ------------ | --------------- | --------------- |
| lead         | 3.600 s         | 3.600 s         |
| buffer       | 4.000 s         | 4.000 s         |
| dropped span | 3.100 s (77.5%) | 3.100 s (77.5%) |
| dropped h²   | **32.3%**       | **0.91%**       |

The dropped span is identical and the energy in it differs 35-fold. At 30 Hz
those early samples lie below the cutoff and are near-silent; at 20 Hz they
carry real signal, and they also enlarge the total the fraction is taken
against.

So here the cutoff changes the _content_ of the dropped span and not its
geometry. **That is a property of this chirp-time bin, not a rule.** LAL rounds
the buffer to a power of two, which happens to absorb the 20-to-30 Hz difference
for 30+25 — and for 25+25, 30+30, 40+30 and 50+50 — but not for lighter systems:

| binary | buffer at 20 Hz | buffer at 30 Hz | dropped h²    |
| ------ | --------------- | --------------- | ------------- |
| 30+25  | 4.000 s         | 4.000 s         | 32.3% / 0.91% |
| 10+10  | 16.000 s        | 8.000 s         | 74.6% / 53.7% |

## The offset dominates

Same binary, LAL, H1. Energy lost at a 20 Hz cutoff against 30 Hz, by how far
`coa_time` sits past the boundary:

| `coa_time` − boundary | 20 Hz | 30 Hz |
| --------------------- | ----- | ----- |
| 1 ms                  | 99.9% | 99.8% |
| 0.1 s                 | 72.8% | 48.7% |
| 0.25 s                | 54.2% | 10.5% |
| 0.5 s                 | 32.3% | 0.91% |
| 1 s                   | 2.9%  | 0.34% |

The gap between cutoffs is widest in the middle, where the dropped span covers
just the band between them.

## A binary neutron star

1.4+1.35 solar masses, LAL, H1. Worse in absolute terms, and here the cutoff
moves the geometry too, because the chirp time dominates the rounding:

|                                        | 20 Hz       | 30 Hz      |
| -------------------------------------- | ----------- | ---------- |
| lead                                   | 230.4 s     | 57.6 s     |
| buffer                                 | 256.0 s     | 64.0 s     |
| dropped h², `coa_time` on the boundary | **99.998%** | **99.93%** |
| dropped h², `coa_time` 0.5 s past it   | **96.1%**   | **93.1%**  |

Those last two rows are the same binary half a second apart. On the boundary the
only thing retained is the near-silent post-merger tail; half a second later the
merger itself is inside the segment. That pair is the reason an offset has to be
given alongside any figure on this page.

The 256 s buffer is specific to 1.4+1.35. A 2.0+1.5 binary leads by about 115 s
at the same cutoff.

## The backend changes the geometry

Leads for the same 30+25 binary at 1024 Hz:

| backend           | 20 Hz   | 30 Hz   |
| ----------------- | ------- | ------- |
| LAL               | 3.600 s | 3.600 s |
| ripple (defaults) | 4.050 s | 2.813 s |

For ripple the cutoff moves the buffer itself, so none of the LAL figures above
transfer to it. PyCBC does not report a length at all — its
`pre_coalescence_duration` returns `None`, meaning _unknown_, and a caller must
not read that as zero.

## What does not matter much

- **Distance** cancels out of a fraction entirely.
- **Detector network** moves it by about a percentage point: 32.9% for a
  three-detector ET triangle against 32.3% for H1, at 20 Hz and 0.5 s past the
  boundary.
- **Sky position, polarization and inclination** together move it by a few
  percentage points: across 54 combinations the 20 Hz / 0.5 s figure ranges from
  29.7% to 33.4%. Larger than the network effect, and named for the same reason
  — to say explicitly that these are _not_ where the orders of magnitude come
  from.
- **GPS epoch** is negligible: 32.32% to 32.34% across four epochs spanning 2020
  to 2030. Worth stating because it is the one knob here that looks like it
  should matter — the antenna pattern rotates with the Earth — and over a 4 s
  buffer it does not.
- **`ringdown_fraction`** between 0.05 and 0.2 leaves the 4.000 s buffer and the
  32.3% unchanged.

## These are a proxy, not an SNR loss

Every figure is unweighted h². A matched-filter SNR loss needs a detector PSD
and frequency-domain weighting, neither of which is applied here. **None of
these numbers is anchored against an external SNR tool** — they are this
package's own measurement of its own output, and should be read as an indication
of scale rather than as a detection-efficiency statement.
