"""Finding the safe travel limits of a jaw servo.

Do this with the servo **disconnected from the jaw linkage**. A servo will
happily drive past what the mechanism allows, and it is stronger than a printed
part or a glued joint. Find the numbers first, connect the linkage second.

    python -m anima.calibrate --pin 18

Sweep with the arrow-key equivalents (``+`` / ``-``), note the angle where the
mouth is shut and the angle where it is open as far as you want it to go, then
put those in config.yaml as ``jaw.closed_angle`` and ``jaw.open_angle``.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="anima.calibrate", description="Find jaw servo limits interactively."
    )
    parser.add_argument("--pin", type=int, default=18, help="BCM pin (default: 18)")
    parser.add_argument("--min", type=float, default=-90.0, help="sweep lower bound")
    parser.add_argument("--max", type=float, default=90.0, help="sweep upper bound")
    parser.add_argument("--step", type=float, default=2.0, help="degrees per keypress")
    args = parser.parse_args(argv)

    try:
        from gpiozero import AngularServo
    except ImportError:
        print(
            "gpiozero is not installed. This command only runs on a Raspberry Pi.\n"
            "  pip install gpiozero pigpio",
            file=sys.stderr,
        )
        return 2

    factory = None
    try:
        from gpiozero.pins.pigpio import PiGPIOFactory

        factory = PiGPIOFactory()
    except Exception:
        print("pigpio not available -- using software PWM (expect jitter).\n")

    servo = AngularServo(
        args.pin,
        min_angle=args.min,
        max_angle=args.max,
        min_pulse_width=0.5 / 1000,
        max_pulse_width=2.5 / 1000,
        pin_factory=factory,
    )

    angle = 0.0
    servo.angle = angle

    print(
        f"Servo on BCM pin {args.pin}. Linkage should be DISCONNECTED.\n"
        f"  +  open further      -  close further\n"
        f"  0  return to zero    q  quit\n"
        f"Step is {args.step} degrees; type a number to jump straight to it.\n"
    )

    try:
        while True:
            try:
                key = input(f"[{angle:+.1f} deg] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if key in {"q", "quit", "exit"}:
                break
            if key == "0":
                angle = 0.0
            elif key.startswith("+"):
                angle += args.step * max(1, len(key))
            elif key.startswith("-"):
                angle -= args.step * max(1, len(key))
            else:
                try:
                    angle = float(key)
                except ValueError:
                    print("  ? use + / - / a number / q")
                    continue

            angle = max(args.min, min(args.max, angle))
            servo.angle = angle
    finally:
        servo.angle = 0.0
        servo.close()
        print("\nServo released. Put your two numbers in config.yaml:")
        print("  jaw.closed_angle:  the angle where the mouth is shut")
        print("  jaw.open_angle:    the angle where it is open as far as you want")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
