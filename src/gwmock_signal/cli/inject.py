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
"""CLI sub-app for gravitational-wave signal injection commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from gwpy.timeseries import TimeSeries, TimeSeriesDict

from gwmock_signal.detector import CustomDetector
from gwmock_signal.network import Network
from gwmock_signal.pipeline import inject_cbc_signal
from gwmock_signal.waveform.backends import GWSignalBackend, LALSimulationBackend, PyCBCBackend, RippleBackend

inject_app = typer.Typer(
    name="inject",
    help="Inject gravitational-wave signals into detector data. "
    "See https://leuven-gravity-institute.github.io/gwmock-signal/ for full documentation.",
)


@inject_app.command("cbc")
def cbc(  # noqa: PLR0912, PLR0913, PLR0915, PLR0917
    params: Annotated[
        Path,
        typer.Option("--params", help="Path to JSON file with CBC injection parameters.", show_default=False),
    ],
    network: Annotated[
        str,
        typer.Option(
            "--network",
            help=(
                "Named network preset (e.g. H1L1V1) or comma-separated detector codes "
                "(e.g. H1,L1,V1). Any code from Network.list_lal_detectors() is accepted."
            ),
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Output file path (GWpy TimeSeriesDict HDF5). If omitted, a one-line summary per detector is printed to stdout.",
        ),
    ] = None,
    sample_rate: Annotated[
        int,
        typer.Option("--sample-rate", help="Sample rate in Hz."),
    ] = 4096,
    f_min: Annotated[
        float,
        typer.Option("--f-min", help="Minimum frequency cutoff in Hz passed to the waveform generator."),
    ] = 20.0,
    duration: Annotated[
        float,
        typer.Option("--duration", help="Length of the zero-noise background segment in seconds."),
    ] = 16.0,
    approximant: Annotated[
        str,
        typer.Option(
            "--approximant",
            help="Time-domain approximant name for the selected --backend "
            "(LAL, PyCBC, and ripple lists differ; see user guide).",
        ),
    ] = "IMRPhenomD",
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help="Waveform backend: 'lal' (default), 'gwsignal', 'pycbc', or 'ripple' (JAX; currently IMRPhenomD only).",
        ),
    ] = "lal",
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help=(
                "Optional integer random seed for reproducibility. "
                "Sets numpy.random.seed before building the zero-noise background."
            ),
        ),
    ] = None,
    earth_rotation: Annotated[
        bool,
        typer.Option(
            "--earth-rotation/--no-earth-rotation",
            help=(
                "Enable time-dependent antenna-pattern evaluation for Earth rotation. "
                "Recommended for signals longer than a few seconds. "
                "Set --no-earth-rotation for short waveforms where rotation is negligible."
            ),
        ),
    ] = True,
    interpolate_if_offset: Annotated[
        bool,
        typer.Option(
            "--interpolate-if-offset/--no-interpolate-if-offset",
            help=(
                "Enable cubic interpolation when the injection start is not on a "
                "target-sample boundary. Use --no-interpolate-if-offset for strict "
                "grid alignment (returns the background unchanged when off-grid)."
            ),
        ),
    ] = True,
) -> None:
    """Inject a single CBC event into zero-noise detector data.

    Loads CBC parameters from a JSON file, resolves the detector network,
    generates a zero-noise background, runs the injection pipeline, and
    either writes the result to an HDF5 file or prints a per-detector summary.

    See https://leuven-gravity-institute.github.io/gwmock-signal/ for full documentation.
    """
    # Load CBC parameters from JSON
    try:
        with params.open() as fh:
            cbc_params = json.load(fh)
    except FileNotFoundError:
        raise typer.BadParameter(f"Parameter file not found: {params}", param_hint="--params") from None
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON in {params}: {exc}", param_hint="--params") from exc

    # Resolve network: file path (existing or recognised extension) uses from_file;
    # otherwise look up the pre-defined catalog.
    network_path = Path(network)
    if network_path.exists() or network_path.suffix.lower() in (".yaml", ".yml", ".json"):
        try:
            net = Network.from_file(network_path)
        except (ValueError, FileNotFoundError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--network") from exc
    else:
        try:
            net = Network.from_name(network)
        except ValueError:
            codes = [c.strip() for c in network.split(",")]
            try:
                net = Network.from_detectors(codes, name=network)
            except ValueError as exc:
                raise typer.BadParameter(
                    f"{exc}\n"
                    f"Named presets: {Network.list_names()}. "
                    "Or pass comma-separated detector codes, e.g. H1,L1,V1.",
                    param_hint="--network",
                ) from exc

    if seed is not None:
        np.random.seed(seed)

    if sample_rate <= 0:
        raise typer.BadParameter("--sample-rate must be > 0", param_hint="--sample-rate")
    if duration <= 0:
        raise typer.BadParameter("--duration must be > 0", param_hint="--duration")
    if f_min <= 0:
        raise typer.BadParameter("--f-min must be > 0", param_hint="--f-min")

    backend_name = backend.strip().lower()
    backend_factories = {
        "lal": LALSimulationBackend,
        "gwsignal": GWSignalBackend,
        "pycbc": PyCBCBackend,
        "ripple": RippleBackend,
    }
    if backend_name not in backend_factories:
        raise typer.BadParameter(
            "--backend must be one of 'lal', 'gwsignal', 'pycbc', or 'ripple'", param_hint="--backend"
        )
    try:
        waveform_backend = backend_factories[backend_name]()
    except ImportError as exc:
        raise typer.BadParameter(str(exc), param_hint="--backend") from exc

    # Build zero-noise background centred on coa_time
    try:
        coa_time = float(cbc_params["coa_time"])
    except KeyError:
        raise typer.BadParameter("Missing required parameter: 'coa_time'", param_hint="--params") from None
    except (TypeError, ValueError):
        raise typer.BadParameter("Parameter 'coa_time' must be a number", param_hint="--params") from None
    cbc_params["coa_time"] = coa_time
    # Normalise to plain string names for background dict keys and output labels.
    det_str_names = [d.name if isinstance(d, CustomDetector) else d for d in net.detector_names]

    n_samples = int(duration * sample_rate)
    background = {
        name: TimeSeries(
            np.zeros(n_samples),
            t0=coa_time - duration / 2,
            sample_rate=sample_rate,
        )
        for name in det_str_names
    }

    # Run the injection pipeline
    result = inject_cbc_signal(
        waveform_model=approximant,
        params=cbc_params,
        detector_names=net.detector_names,
        background=background,
        sampling_frequency=float(sample_rate),
        minimum_frequency=f_min,
        waveform_backend=waveform_backend,
        earth_rotation=earth_rotation,
        interpolate_if_offset=interpolate_if_offset,
    )

    if output is not None:
        tsd = TimeSeriesDict(result.to_dict())
        tsd.write(str(output), format="hdf5")
    else:
        for name in det_str_names:
            ts = result[name]
            rms = float(np.sqrt(np.mean(ts.value**2)))
            typer.echo(f"{name}  rms={rms:.4e}  duration={ts.duration.value:.1f}s")
