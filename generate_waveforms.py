#!/usr/bin/env python3
"""
Pre-compute waveform peak data for all audio files in the demo folder.
Outputs waveforms.js which contains min/max pairs per pixel column (800 cols).

Usage:
  python generate_waveforms.py [--width 800]

Run this after prepare_demo_audio.py. The generated waveforms.js should be
placed next to index.html.
"""

import os
import json
import argparse
import numpy as np
import soundfile as sf

AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waveforms.js")
WIDTH = 800  # number of columns (pixels)


def compute_peaks(filepath, width):
    """Read audio file, return list of [min, max] pairs downsampled to `width` columns."""
    try:
        data, sr = sf.read(filepath, dtype='float32', always_2d=True)
    except Exception as e:
        print(f"  ✗ Cannot read: {filepath} ({e})")
        return None

    # Mix to mono
    mono = data.mean(axis=1)
    n = len(mono)
    step = n / width
    peaks = []
    for i in range(width):
        start = int(i * step)
        end = int((i + 1) * step)
        chunk = mono[start:end]
        if len(chunk) == 0:
            peaks.append([0.0, 0.0])
        else:
            peaks.append([round(float(chunk.min()), 4), round(float(chunk.max()), 4)])
    return peaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=WIDTH)
    args = parser.parse_args()

    if not os.path.isdir(AUDIO_DIR):
        print(f"Audio directory not found: {AUDIO_DIR}")
        return

    waveform_data = {}  # key: "song/model/stem.wav" → peaks

    for song in sorted(os.listdir(AUDIO_DIR)):
        song_path = os.path.join(AUDIO_DIR, song)
        if not os.path.isdir(song_path):
            continue
        print(f"\n[{song}]")
        for model in sorted(os.listdir(song_path)):
            model_path = os.path.join(song_path, model)
            if not os.path.isdir(model_path):
                continue
            for wav_file in sorted(os.listdir(model_path)):
                if not wav_file.endswith('.wav'):
                    continue
                filepath = os.path.join(model_path, wav_file)
                # Key matches what JS expects: "song/model/stem.wav"
                key = f"{song}/{model}/{wav_file}"
                print(f"  {model}/{wav_file} ...", end=" ")
                peaks = compute_peaks(filepath, args.width)
                if peaks:
                    waveform_data[key] = peaks
                    print("✓")

    # Write as JS module
    print(f"\nWriting {OUTPUT_FILE} ({len(waveform_data)} waveforms)...")
    with open(OUTPUT_FILE, 'w') as f:
        f.write("// Auto-generated waveform peak data. Do not edit.\n")
        f.write(f"// Width: {args.width} columns per waveform.\n")
        f.write("const WAVEFORM_DATA = ")
        json.dump(waveform_data, f, separators=(',', ':'))
        f.write(";\n")

    print("✅ Done!")


if __name__ == "__main__":
    main()
