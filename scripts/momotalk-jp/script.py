from __future__ import annotations

import logging
import os
import sys
import threading

# Enable DEBUG logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from engine.runtime import (
    action,
    cleanup,
    hold_key,
    hotkey_callback,
    hotkey_restart,
    hotkey_stop,
    press_until_or_skip_on_stable,
    press_until_stable,
    reset_stop,
    sleep,
    wait_for,
    wait_for_restart,
    wait_for_stable,
)

# MARKER1_NEAR = (270, 226)
# MARKER1_RADIUS = 250
MARKER1_NEAR = (220, 220)
MARKER1_RADIUS = 300
MARKER2_NEAR = (348,184)
MARKER2_RADIUS = 250
STATECHECK2_NEAR = (1733,65)
STATECHECK2_RADIUS = 300

def run_sequence() -> None:
    action("marker1.png", click_duration=0.5, near=MARKER1_NEAR, radius=MARKER1_RADIUS)
    wait_for("statecheck4.png", 30.0, roi_ref=(212, 150, 317, 62), verify_pixels=[(43, 4, (252, 141, 162), 30)])
    action("marker2.png")
    # action("marker3.png", (-50, 0))
    # action("marker4.png", (-50, 70))
    # action("marker5.png", (-50, 50))
    for i in range(5):
        print('Iteration', i + 1)
        action("marker6.png", (-80, 80 + (120 * i)))
        # Check if press_until succeeded
        found = press_until_or_skip_on_stable(
            "statecheck2a.png",
            ["1", "space"],
            60.0,
            step_delay_s=0.5,
            roi_ref=(1695, 924, 150, 112),  # x, y, w, h
            verify_pixels=[
                (1810, 960, (44, 69, 99), 5, {"global": True, "sample": 2}),
            ],
            stable_count=8,
            poll_interval_s=0.5,
            wait_for_change=True,
            require=False,
        )
        if not found: # Only continue if match was found
            print(f'Iteration {i + 1}: statecheck2 not found, skipping to next iteration')
            continue  # Skip rest of loop, go to next iteration
        print(f'Iteration {i + 1}: statecheck2 found, proceeding...') # Match found - proceed with the rest
        hold_key("esc", 2)
        sleep(0.5)
        hold_key("space",2 , repeat= 3)
        sleep(3)
        press_until_stable(
            ["1", "space"],
            roi_ref=(1038, 222, 672, 716),  # TODO: calibrate with stability_tester
            timeout_s=60.0,
            poll_interval_s=0.5,
            stable_count=40,
            attempt_timeout_s=40.0,
            press_while_waiting=True,
            require=False,
        )
    hold_key("esc", 2)


def _toggle_loop(loop_flag: threading.Event) -> None:
    if loop_flag.is_set():
        loop_flag.clear()
        logging.info("Looping OFF")
    else:
        loop_flag.set()
        logging.info("Looping ON")


def main() -> None:
    loop_enabled = threading.Event()
    hotkey_stop("F6")
    hotkey_restart("F5")
    hotkey_callback("F4", lambda: _toggle_loop(loop_enabled), name="loop_toggle")
    while True:
        wait_for_restart()
        while True:
            reset_stop()
            try:
                run_sequence()
            except SystemExit:
                break
            if not loop_enabled.is_set():
                break

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
