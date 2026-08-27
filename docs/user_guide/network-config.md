---
title: Network configuration files
description:
    YAML/JSON schema for defining detector networks with Network.from_file,
    including custom detector geometries.
icon: material/file-cog
---

# Network configuration files

[`Network.from_file`](../api/network/) reads a YAML or JSON file that defines a
detector network. The file format supports both **simple detector-code lists**
(when every detector is pre-defined in LAL) and **full custom-detector
geometries** for non-standard observatories.

This page shows the file schema and working examples. For programmatic
construction without a file, use [`Network.from_detectors`](../api/network/)
directly.

## File schema

| Key         | Type             | Required | Description                         |
| ----------- | ---------------- | -------- | ----------------------------------- |
| `name`      | `string`         | Yes      | Human-readable network label.       |
| `detectors` | `list[detector]` | Yes      | Non-empty list of detector objects. |

Each **detector object** is a mapping:

| Key                | Type     | Required | Description                                 |
| ------------------ | -------- | -------- | ------------------------------------------- |
| `name`             | `string` | Yes      | Detector identifier (e.g. `"H1"`, `"ET1"`). |
| `prefix`           | `string` | No       | Two-character LAL prefix (auto-generated).  |
| `latitude_deg`     | `float`  | \*       | Geodetic latitude in degrees.               |
| `latitude_rad`     | `float`  | \*       | Geodetic latitude in radians.               |
| `longitude_deg`    | `float`  | \*       | Geodetic longitude in degrees.              |
| `longitude_rad`    | `float`  | \*       | Geodetic longitude in radians.              |
| `elevation_m`      | `float`  | \*       | Elevation above WGS-84 ellipsoid in metres. |
| `xarm_azimuth_deg` | `float`  | \*       | X-arm azimuth in degrees.                   |
| `xarm_azimuth_rad` | `float`  | \*       | X-arm azimuth in radians.                   |
| `yarm_azimuth_deg` | `float`  | \*       | Y-arm azimuth in degrees.                   |
| `yarm_azimuth_rad` | `float`  | \*       | Y-arm azimuth in radians.                   |
| `xarm_tilt_deg`    | `float`  | No       | X-arm tilt in degrees (default `0.0`).      |
| `xarm_tilt_rad`    | `float`  | No       | X-arm tilt in radians (default `0.0`).      |
| `yarm_tilt_deg`    | `float`  | No       | Y-arm tilt in degrees (default `0.0`).      |
| `yarm_tilt_rad`    | `float`  | No       | Y-arm tilt in radians (default `0.0`).      |

\* **Required only when defining a custom geometry.** If none of the geometry
keys are present the entry is treated as a built-in LAL detector code and the
name must match a code returned by
[`Network.list_lal_detectors`](../api/network/).

!!! warning "Angle conventions" For each angle parameter, provide **either** the
`_deg` **or** the `_rad` variant — never both. Providing both raises a
`ValueError`.

---

## Example 1 — Built-in detector codes only

For standard LIGO-Virgo-KAGRA configurations, every detector name is a known LAL
code. The file is minimal:

```yaml
# hlvk.yaml — LIGO Hanford + Livingston + Virgo + KAGRA
name: HLVK
detectors:
    - name: H1
    - name: L1
    - name: V1
    - name: K1
```

Load it:

```python
from gwmock_signal.network import Network

net = Network.from_file("hlvk.yaml")
print(net.name)  # HLVK
print(net.detector_names)  # ('H1', 'L1', 'V1', 'K1')
```

You can also use `--network hlvk.yaml` in the CLI (see
[Command-line interface](cli.md)).

---

## Example 2 — A single custom detector (degrees)

Define a fabricated site with full geometry in degrees:

```yaml
# custom_observatory.yaml
name: Custom Observatory
detectors:
    - name: CUST1
      latitude_deg: 45.0
      longitude_deg: 10.0
      elevation_m: 100.0
      xarm_azimuth_deg: 90.0
      yarm_azimuth_deg: 0.0
```

Load it:

```python
from gwmock_signal.network import Network

net = Network.from_file("custom_observatory.yaml")
# net.detector_names[0] is a CustomDetector instance
det = net.detector_names[0]
print(det.name, det.latitude_rad)
```

---

## Example 3 — Full custom detector with radians and tilt

For precision work, supply angles in radians directly:

```yaml
# triangle_site.yaml
name: Triangle Site
detectors:
    - name: TD1
      prefix: 'T1'
      latitude_rad: 0.698
      longitude_rad: 0.209
      elevation_m: 200.0
      xarm_azimuth_rad: 1.234
      yarm_azimuth_rad: 2.345
      xarm_tilt_deg: 0.01
      yarm_tilt_rad: 0.0005
    - name: TD2
      latitude_rad: 0.699
      longitude_rad: 0.210
      elevation_m: 195.0
      xarm_azimuth_rad: 3.456
      yarm_azimuth_rad: 4.567
```

You can mix `_deg` and `_rad` variants across parameters (but not for the same
parameter in the same entry).

---

## Example 4 — Mixed built-in and custom detectors

Combine known LAL codes with custom sites in the same network:

```yaml
# mixed_network.yaml
name: HLVK + Custom
detectors:
    - name: H1
    - name: L1
    - name: V1
    - name: K1
    - name: CD1
      latitude_deg: 35.0
      longitude_deg: 140.0
      elevation_m: 300.0
      xarm_azimuth_deg: 45.0
      yarm_azimuth_deg: 135.0
```

---

## Bundled presets

Several detector networks are **bundled** with the package under
`gwmock_signal.data.detectors`:

| Preset name            | File                                |
| ---------------------- | ----------------------------------- |
| `ET-Triangle-Sardinia` | `et-triangle-sardinia.yaml`         |
| `ET-Sardinia`          | _(alias of `ET-Triangle-Sardinia`)_ |
| `ET-Triangle-EMR`      | `et-triangle-emr.yaml`              |
| `ET-EMR`               | _(alias of `ET-Triangle-EMR`)_      |
| `ET-2L-Aligned`        | `et-2l-aligned.yaml`                |
| `ET-2L-Misaligned`     | `et-2l-misaligned.yaml`             |

Load a bundled preset programmatically:

```python
from gwmock_signal.network import Network

net = Network.from_preset("ET-Triangle-Sardinia")
```

---

## `.interferometer` migration

Legacy Bilby `.interferometer` files are **deprecated** and will be removed in a
future major release. Use `interferometer_config_to_custom_detector` to migrate:

```python
from gwmock_signal.io import interferometer_config_to_custom_detector
from gwmock_signal.network import Network

detector = interferometer_config_to_custom_detector("legacy_site.interferometer")
net = Network.from_detectors([detector], name="Migrated Site")
```

Then save the equivalent YAML with `Network.from_file` so you can drop the
`.interferometer` file:

```yaml
name: Migrated Site
detectors:
    - name: LEGACY1
      latitude_deg: 40.0
      longitude_deg: 9.0
      elevation_m: 50.0
      xarm_azimuth_deg: 70.0
      yarm_azimuth_deg: 130.0
```

---

## Runtime detector codes

To see every detector code known to your installed LAL:

```python
from gwmock_signal.network import Network

print(Network.list_lal_detectors())
```

These are the codes you can use as bare `name` entries in a network file.

---

## See also

- [Network API reference](../api/network/)
- [Custom detectors](custom-detectors.md)
- [Detector projection examples](detector-projection.md)
- [Command-line interface](cli.md)
- [User guide overview](index.md)
