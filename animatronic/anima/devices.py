"""Listing audio devices, so you can pin the right microphone in config.

    python -m anima.devices

The default input device is rarely the one you want on a Pi with a USB mic and
an HDMI display both claiming to be audio hardware. Find the index here and set
it as ``ears.whisper.device_index``.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    try:
        import sounddevice
    except Exception as exc:
        print(f"sounddevice unavailable: {exc}", file=sys.stderr)
        print(
            "\nInstall it with:\n"
            "  sudo apt install libportaudio2\n"
            "  pip install sounddevice",
            file=sys.stderr,
        )
        return 2

    try:
        default_in, default_out = sounddevice.default.device
    except Exception:
        default_in = default_out = None

    print(f"{'idx':>4}  {'in':>3} {'out':>3}  {'rate':>7}  name")
    print("-" * 64)
    for index, device in enumerate(sounddevice.query_devices()):
        marks = ""
        if index == default_in:
            marks += " <- default in"
        if index == default_out:
            marks += " <- default out"
        print(
            f"{index:>4}  {device['max_input_channels']:>3} "
            f"{device['max_output_channels']:>3}  "
            f"{int(device['default_samplerate']):>7}  {device['name']}{marks}"
        )

    print("\nSet ears.whisper.device_index in config.yaml to an index with inputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
