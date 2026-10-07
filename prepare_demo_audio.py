#!/usr/bin/env python3
"""
生成 demo page 音频文件夹结构。
指定一首歌名，从各模型源文件夹裁剪指定时间段，无损保存。

用法:
  python prepare_demo_audio.py --song "Al James - Schoolboy Facination" --start 30 --end 50
  python prepare_demo_audio.py --song "Angels In Amplifiers - I'm Alright" --start 70 --end 80
  python prepare_demo_audio.py --song "Bobby Nobody - Stitch Up" --start 120 --end 130
  python prepare_demo_audio.py --song "Carlos Gonzalez - A Place For Us" --start 90 --end 100
  python prepare_demo_audio.py --song "Georgia Wonder - Siren" --start 300 --end 310
  python prepare_demo_audio.py --song "Lyndsey Ollard - Catching Up" --start 150 --end 160
  python prepare_demo_audio.py --song "PR - Oh No" --start 30 --end 40
"""

import argparse
import os
import soundfile as sf
import numpy as np

# ============================================================
# 配置：每个模型的 VBDO 源文件夹
# 文件夹内音频命名格式: {歌名}.wav
# ============================================================

SONG_SUFFIX = ".wav"

# GT stems + mixture
GT_SOURCES = {
    "mixture": "/mnt/dataset/MUSDB18HQ/test/{song}/mixture.wav",
    "vocals":  "/mnt/dataset/MUSDB18HQ/test/{song}/vocals.wav",
    "bass":    "/mnt/dataset/MUSDB18HQ/test/{song}/bass.wav",
    "drums":   "/mnt/dataset/MUSDB18HQ/test/{song}/drums.wav",
    "other":   "/mnt/dataset/MUSDB18HQ/test/{song}/other.wav",
}

# 模型分离结果路径
# 每个模型 → dict: stem → 文件夹路径（文件夹下文件名为 {歌名}.wav）
MODEL_SOURCES = {
    "htdemucs": {
        "vocals": "/mnt/gaia_result/mss/htdemucs/vocals",
        "bass":   "/mnt/gaia_result/mss/htdemucs/bass",
        "drums":  "/mnt/gaia_result/mss/htdemucs/drums",
        "other":  "/mnt/gaia_result/mss/htdemucs/other",
    },
    "openunmix": {
        "vocals": "/mnt/gaia_result/mss/openunmix/vocals",
        "bass":   "/mnt/gaia_result/mss/openunmix/bass",
        "drums":  "/mnt/gaia_result/mss/openunmix/drums",
        "other":  "/mnt/gaia_result/mss/openunmix/other",
    },
    # "FLOMSS-S-500k": {
    #     "vocals": "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_vocals_48k-remix-xloss-3705012-c4-3809668-3812486/refiner_training/version_0/test/pred",
    #     "bass":   "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_bass_48k-remix-xloss-3760024-c4-3809941-3812487/refiner_training/version_0/test/pred",
    #     "drums":  "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_drums_48k-remix-xloss-3754598-c4-3810913-3812520/refiner_training/version_0/test/pred",
    #     "other":  "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_other_48k-remix-xloss-3759695-c4-3812407-3813985/refiner_training/version_0/test/pred",
    # },
    "FLOMSS-M": {
        "vocals": "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_vocals_48k-remix-xloss-sgmsvs-old-3840460-3843200/pred",
        "bass":   "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_bass_48k-remix-xloss-sgmsvs-old-3843210-3844467/pred",
        "drums":  "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_drums_48k-remix-xloss-sgmsvs-old-3844466-3846755/pred",
        "other":  "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_other_48k-remix-xloss-sgmsvs-old-3846747-3847970/pred",
    },
    "instglow": {
        "vocals": "/mnt/gaia_result/mss/instglow/vocals",
        "bass":   "/mnt/gaia_result/mss/instglow/bass",
        "drums":  "/mnt/gaia_result/mss/instglow/drums",
        "other":  "/mnt/gaia_result/mss/instglow/other",
    },
    # "flowsep": {
    #     "vocals": "/mnt/gaia_result/mss/flowsep/vocals",
    #     "bass":   "/mnt/gaia_result/mss/flowsep/bass",
    #     "drums":  "/mnt/gaia_result/mss/flowsep/drums",
    #     "other":  "/mnt/gaia_result/mss/flowsep/other",
    # },
    "mge_ldm": {
        "vocals": "/mnt/gaia_result/mss/mge_ldm/vocals",
        "bass":   "/mnt/gaia_result/mss/mge_ldm/bass",
        "drums":  "/mnt/gaia_result/mss/mge_ldm/drums",
        #"other":  "/mnt/gaia_result/mss/mge_ldm/other",
    },
    "sam_audio": {
        "vocals": "/mnt/gaia_result/mss/sam_audio_small/vocals",
        "bass":   "/mnt/gaia_result/mss/sam_audio_small/bass",
        "drums":  "/mnt/gaia_result/mss/sam_audio_small/drums",
        #"other":  "/mnt/gaia_result/mss/sam_audio_small/other",
    },
    "FLOMSS-S": {
        "vocals": "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_vocals_48k-remix-xloss-3705012-3712122/refiner_training/version_0/test/pred",
        "bass":   "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_bass_48k-remix-xloss-3760024-3761910/refiner_training/version_0/test/pred",
        "drums":  "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_drums_48k-remix-xloss-3754598-3760301/refiner_training/version_0/test/pred",
        "other":  "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_other_48k-remix-xloss-3759695-3760492/refiner_training/version_0/test/pred",
    },
    "SCNet": {
        "vocals": "/mnt/gaia_result/mss/scnet/vocals",
        "bass":   "/mnt/gaia_result/mss/scnet/bass",
        "drums":  "/mnt/gaia_result/mss/scnet/drums",
        "other":  "/mnt/gaia_result/mss/scnet/other",
    },
    "FLOMSS-Disc.": {
        "vocals": "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_vocals_48k-remix-xloss-predictive-3721085-3725698/refiner_training/version_0/test/pred",
        "bass":   "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_bass_48k-remix-xloss-predictive-3767560-3770301/refiner_training/version_0/test/pred",
        "drums":  "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_drums_48k-remix-xloss-predictive-3770297-3775223/refiner_training/version_0/test/pred",
        "other":  "/mnt/gaia_result/mss/cfm/eval/a05_b015_sy_05_x_5_other_48k-remix-xloss-predictive-3770300-3775229/refiner_training/version_0/test/pred",
    }
}

# 输出根目录
OUTPUT_ROOT = "/mnt/flomss/audio"

STEMS = ["vocals", "bass", "drums", "other"]


def crop_and_save(src_path: str, dst_path: str, start_sec: float, end_sec: float):
    """
    无损裁剪：用 soundfile 读取原始 PCM 样本，裁剪后以相同格式/采样率保存。
    不做任何重采样或编解码。
    """
    info = sf.info(src_path)
    sr = info.samplerate
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr) if end_sec > 0 else info.frames

    # 确保不越界
    start_sample = max(0, min(start_sample, info.frames))
    end_sample = max(start_sample, min(end_sample, info.frames))

    # 读取指定范围
    data, _ = sf.read(src_path, start=start_sample, stop=end_sample, dtype='float64', always_2d=True)

    # 保存为相同格式（WAV PCM）
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sf.write(dst_path, data, sr, subtype=info.subtype, format=info.format)
    print(f"  ✓ {dst_path} ({data.shape[0]} samples, {data.shape[1]}ch, {sr}Hz)")


def process_song(song: str, start_sec: float, end_sec: float):
    print(f"\n{'='*60}")
    print(f"Processing: {song}")
    print(f"Time range: {start_sec}s — {end_sec}s")
    print(f"{'='*60}")

    song_dir = os.path.join(OUTPUT_ROOT, song)

    # 1. GT (mixture + stems)
    print("\n[GT]")
    for stem, path_template in GT_SOURCES.items():
        src = path_template.format(song=song)
        if not os.path.exists(src):
            print(f"  ✗ NOT FOUND: {src}")
            continue
        dst = os.path.join(song_dir, "gt", f"{stem}.wav")
        crop_and_save(src, dst, start_sec, end_sec)

    # 2. Models
    for model_name, stem_dirs in MODEL_SOURCES.items():
        print(f"\n[{model_name}]")
        for stem in STEMS:
            src_dir = stem_dirs.get(stem)
            if not src_dir:
                print(f"  ✗ No path configured for {stem}")
                continue
            src = os.path.join(src_dir, f"{song}{SONG_SUFFIX}")
            if not os.path.exists(src):
                print(f"  ✗ NOT FOUND: {src}")
                continue
            dst = os.path.join(song_dir, model_name, f"{stem}.wav")
            crop_and_save(src, dst, start_sec, end_sec)

    print(f"\n✅ Done! Output: {song_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare demo audio for one song")
    parser.add_argument("--song", type=str, required=True,
                        help="Song name (folder name in MUSDB18HQ test)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="Start time in seconds (default: 0)")
    parser.add_argument("--end", type=float, default=0.0,
                        help="End time in seconds (default: 0 = full song)")
    args = parser.parse_args()

    process_song(args.song, args.start, args.end)


if __name__ == "__main__":
    main()
