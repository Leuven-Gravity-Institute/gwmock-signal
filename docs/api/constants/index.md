---
title: Constants
description: Unit-conversion constants, checked bit-for-bit against LAL.
icon: material/scale-balance
---

<!-- prettier-ignore-start -->

::: gwmock_signal.constants
    options:
        show_root_heading: true
        heading_level: 2
        show_if_no_docstring: false
        docstring_style: google
        show_source: true
        members:
            - MSUN_SI
            - MTSUN_SI
            - PC_SI
            - MPC_SI

<!-- prettier-ignore-end -->

| Name       | Meaning                                   | Unit | Reference                                    |
| ---------- | ----------------------------------------- | ---- | -------------------------------------------- |
| `MSUN_SI`  | Nominal solar mass                        | kg   | `lal.MSUN_SI`                                |
| `MTSUN_SI` | Solar mass in time units, `G M_sun / c^3` | s    | `lal.MTSUN_SI`                               |
| `PC_SI`    | Parsec                                    | m    | `lal.PC_SI`                                  |
| `MPC_SI`   | Megaparsec, `PC_SI * 1e6`                 | m    | none in LAL; equals `ripplegw.constants.MPC` |

Each value equals its reference bit-for-bit, and the test suite checks this
whenever `lalsuite` or `ripplegw` is installed.
