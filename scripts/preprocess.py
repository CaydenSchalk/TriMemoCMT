import argparse
import json
import glob
import logging
import os
import pickle
import random
import subprocess
import pandas as pd
import soundfile as sf
import tqdm
import numpy as np
import torch

# having issues importing on the HPC
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

from sklearn.model_selection import train_test_split
from pathlib import Path


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

LABEL_MAP = {
    "ang": 0,
    "hap": 1,
    "sad": 2,
    "neu": 3,
}

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_pickle(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def relpath_str(path, start):
    return str(Path(path).resolve().relative_to(Path(start).resolve()))

def export_mp4_to_audio(
    mp4_file: str,
    wav_file: str,
    verbose: bool = False,
):
    """Convert mp4 file to wav file

    Args:
        mp4_file (str): Path to mp4 input file
        wav_file (str): Path to wav output file
        verbose (bool, optional): Whether to print ffmpeg output. Defaults to False.
    """
    try:
        video = VideoFileClip(mp4_file)
    except:
        logging.warning(f"Failed to load {mp4_file}")
        return 0
    audio = video.audio
    audio.write_audiofile(wav_file, verbose=verbose)
    return 1

def preprocess_IEMOCAP(args):
    data_root = args.data_root
    ignore_length = args.ignore_length

    session_id = list(range(1, 6))

    samples = []
    labels = []
    iemocap2label = LABEL_MAP.copy()
    iemocap2label.update({"exc": 1})
    output_root = os.path.join(args.output_root, args.dataset)
    video_output_root = os.path.join(output_root, "video_clips")

    for sess_id in tqdm.tqdm(session_id):
        sess_path = os.path.join(data_root, f"Session{sess_id}")
        sess_audio_root = os.path.join(sess_path, "sentences", "wav")
        sess_text_root = os.path.join(sess_path, "dialog", "transcriptions")
        sess_label_root = os.path.join(sess_path, "dialog", "EmoEvaluation")
        sess_video_root = os.path.join(sess_path, "dialog", "avi", "DivX")

        label_paths = glob.glob(os.path.join(sess_label_root, "*.txt"))

        for l_path in label_paths:
            l_name = os.path.basename(l_path)
            transcripts_path = os.path.join(sess_text_root, l_name)

            with open(transcripts_path, "r") as f:
                transcripts = f.readlines()
                transcripts = {
                    t.split(":")[0]: t.split(":")[1].strip()
                    for t in transcripts
                }

            with open(l_path, "r") as f:
                label_lines = f.read().split("\n")

            for l in label_lines:
                if not str(l).startswith("["):
                    continue

                data = l[1:].split()
                utt_id = data[3]
                dialog_id = utt_id[:-5]
                emo = data[4]
                start_time = float(data[0])
                end_time = float(data[2][:-1])

                wav_path = os.path.join(sess_audio_root, dialog_id, f"{utt_id}.wav")
                dialog_video_path = os.path.join(sess_video_root, f"{dialog_id}.avi")
                clip_output_dir = os.path.join(
                    video_output_root, f"Session{sess_id}"
                )
                video_path = os.path.join(clip_output_dir, f"{utt_id}.mp4")

                if not os.path.exists(dialog_video_path):
                    logging.warning(f"Missing video file: {dialog_video_path}")
                    continue

                wav_data, _ = sf.read(wav_path, dtype="int16")
                if len(wav_data) < ignore_length:
                    logging.warning(
                        f"Ignoring sample {wav_path} with length {len(wav_data)}"
                    )
                    continue

                emo = iemocap2label.get(emo, None)
                if emo is None:
                    continue

                os.makedirs(clip_output_dir, exist_ok=True)
                if not os.path.exists(video_path):
                    try:
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-y",
                                "-ss",
                                str(start_time),
                                "-i",
                                dialog_video_path,
                                "-t",
                                str(end_time - start_time),
                                "-c:v",
                                "libx264",
                                "-c:a",
                                "aac",
                                "-loglevel",
                                "error",
                                video_path,
                            ],
                            capture_output=True,
                            check=True,
                        )
                    except Exception as e:
                        logging.warning(
                            f"Can not clip video data: {dialog_video_path}\nException: {e}"
                        )
                        continue

                text_query = utt_id + " [{:08.4f}-{:08.4f}]".format(
                    start_time, end_time
                )
                text = transcripts.get(text_query, None)

                if text is None:
                    text_query = utt_id + " [{:08.4f}-{:08.4f}]".format(
                        start_time, end_time + 0.0001
                    )
                    text = transcripts.get(text_query, None)

                if text is None:
                    text_query = utt_id + " [{:08.4f}-{:08.4f}]".format(
                        start_time + 0.0001, end_time
                    )
                    text = transcripts.get(text_query, None)

                if text is None:
                    logging.warning(f"Missing transcript for utterance: {utt_id}")
                    continue

                video_rel = os.path.relpath(video_path, output_root)
                audio_rel = os.path.relpath(wav_path, args.data_root)

                samples.append((video_rel, audio_rel, text, emo))
                labels.append(emo)

    temp = list(zip(samples, labels))
    random.Random(args.seed).shuffle(temp)
    samples, labels = zip(*temp)

    train, test_samples, train_labels, _ = train_test_split(
        samples, labels, test_size=0.1, random_state=args.seed
    )
    train_samples, val_samples, _, _ = train_test_split(
        train, train_labels, test_size=0.1, random_state=args.seed
    )

    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(output_root, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(output_root, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train samples: {len(train_samples)}")
    logging.info(f"Val samples: {len(val_samples)}")
    logging.info(f"Test samples: {len(test_samples)}")
    logging.info(f"Saved to {output_root}")
    logging.info("Preprocessing finished successfully")

def preprocess_ESD(args):
    esd2label = {
        "Angry": "ang",
        "Happy": "hap",
        "Neutral": "neu",
        "Sad": "sad",
    }

    directory = glob.glob(args.data_root + "/*")
    samples = []
    labels = []

    # Loop through all folders
    for dir in tqdm.tqdm(directory):
        # Read label file
        label_path = os.path.join(dir, dir.split("/")[-1] + ".txt")
        with open(label_path, "r") as f:
            label = f.read().strip().splitlines()
        # Extract samples from label file
        for l in label:
            filename, transcript, emotion = l.split("\t")
            target = esd2label.get(emotion, None)
            if target is not None:
                audio_path = os.path.join(dir, emotion, filename + ".wav")
                audio_rel = os.path.relpath(audio_path, args.data_root)

                samples.append(
                    (None, audio_rel, transcript, LABEL_MAP[target])
                )
                # Labels are use for splitting
                labels.append(LABEL_MAP[target])

    # Shuffle and split
    temp = list(zip(samples, labels))
    random.Random(args.seed).shuffle(temp)
    samples, labels = zip(*temp)
    train, test_samples, train_labels, _ = train_test_split(
        samples, labels, test_size=0.2, random_state=args.seed
    )
    train_samples, val_samples, _, _ = train_test_split(
        train, train_labels, test_size=0.1, random_state=args.seed
    )

    # Save data
    output_root = os.path.join(args.output_root, args.dataset)
    with open(os.path.join(output_root, "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(output_root, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(output_root, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train samples: {len(train_samples)}")
    logging.info(f"Train samples: {len(val_samples)}")
    logging.info(f"Test samples: {len(test_samples)}")
    logging.info(f"Saved to {args.dataset + '_preprocessed'}")
    logging.info("Preprocessing finished successfully")


def preprocess_MELD(args):
    meld2label = {
        "anger": "ang",
        "joy": "hap",
        "neutral": "neu",
        "sadness": "sad",
    }

    label_map = LABEL_MAP.copy()

    output_root = os.path.join(args.output_root, args.dataset)
    video_output_root = os.path.join(output_root, "video_clips")
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(video_output_root, exist_ok=True)

    train_csv = os.path.join(args.data_root, "train_sent_emo.csv")
    val_csv = os.path.join(args.data_root, "dev_sent_emo.csv")
    test_csv = os.path.join(args.data_root, "test_sent_emo.csv")

    train_dataframe = pd.read_csv(train_csv)
    val_dataframe = pd.read_csv(val_csv)
    test_dataframe = pd.read_csv(test_csv)

    if args.all_classes:
        meld2label = {}
        label_map = {}
        labels = sorted(list(set(str(row.Emotion) for _, row in test_dataframe.iterrows())))
        for i, label_name in enumerate(labels):
            meld2label[label_name] = i
            label_map[i] = i

        with open(os.path.join(output_root, "classes.json"), "w") as f:
            json.dump(meld2label, f)

    def _preprocess_data(split_name, data_path, dataframe):
        samples = []

        split_video_root = os.path.join(video_output_root, split_name)
        os.makedirs(split_video_root, exist_ok=True)

        for _, row in tqdm.tqdm(dataframe.iterrows(), total=len(dataframe)):
            label = meld2label.get(row.Emotion, None)
            if label is None:
                continue

            transcript = str(row.Utterance)

            filename = f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}"
            raw_video_path = os.path.join(data_path, f"{filename}.mp4")

            # keep audio relative to args.data_root, same as IEMOCAP
            raw_audio_path = os.path.join(data_path, f"{filename}.wav")

            # keep video relative to output_root, same as IEMOCAP
            processed_video_path = os.path.join(split_video_root, f"{filename}.mp4")

            if not os.path.exists(raw_video_path):
                logging.warning(f"Missing video file: {raw_video_path}")
                continue

            # copy video into processed area if it is not there yet
            if not os.path.exists(processed_video_path):
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            raw_video_path,
                            "-c:v",
                            "libx264",
                            "-c:a",
                            "aac",
                            "-loglevel",
                            "error",
                            processed_video_path,
                        ],
                        capture_output=True,
                        check=True,
                    )
                except Exception as e:
                    logging.warning(
                        f"Can not copy/process video data: {raw_video_path}\nException: {e}"
                    )
                    continue

            # extract audio next to raw video so audio_rel is relative to args.data_root
            if not os.path.exists(raw_audio_path):
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            raw_video_path,
                            "-vn",
                            "-acodec",
                            "pcm_s16le",
                            "-ar",
                            "16000",
                            "-ac",
                            "1",
                            "-loglevel",
                            "error",
                            raw_audio_path,
                        ],
                        capture_output=True,
                        check=True,
                    )
                except Exception as e:
                    logging.warning(
                        f"Can not extract audio data: {raw_video_path}\nException: {e}"
                    )
                    continue

            video_rel = os.path.relpath(processed_video_path, output_root)
            audio_rel = os.path.relpath(raw_audio_path, args.data_root)

            samples.append((video_rel, audio_rel, transcript, label_map[label]))

        return samples

    train_samples = _preprocess_data(
        "train",
        os.path.join(args.data_root, "train_splits"),
        train_dataframe,
    )
    val_samples = _preprocess_data(
        "val",
        os.path.join(args.data_root, "dev_splits_complete"),
        val_dataframe,
    )
    test_samples = _preprocess_data(
        "test",
        os.path.join(args.data_root, "output_repeated_splits_test"),
        test_dataframe,
    )

    with open(os.path.join(output_root, "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(output_root, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(output_root, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train samples: {len(train_samples)}")
    logging.info(f"Val samples: {len(val_samples)}")
    logging.info(f"Test samples: {len(test_samples)}")
    logging.info(f"Saved to {output_root}")
    logging.info("Preprocessing finished successfully")


def main(args):
    preprocess_fn = {
        "IEMOCAP": preprocess_IEMOCAP,
        "ESD": preprocess_ESD,
        "MELD": preprocess_MELD,
    }

    preprocess_fn[args.dataset](args)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ds", "--dataset", type=str, default="ESD", choices=["IEMOCAP", "ESD", "MELD"]
    )
    parser.add_argument(
        "-dr",
        "--data_root",
        type=str,
        required=True,
        help="Path to raw dataset root",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Path to processed dataset root",
    )
    parser.add_argument("--all_classes", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ignore_length",
        type=int,
        default=0,
        help="Ignore samples with length < ignore_length",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
