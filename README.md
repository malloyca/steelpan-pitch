# steelpan-pitch

Steelpan-pitch is a low-latency monophonic pitch tracker designed for use with
steelpan audio. It is based on CREPE, a convolutional neural net pitch tracker
that analyzes the time-domain audio signal. Steelpan-pitch is trained
specifically on my steelpan for better accuracy and modifies the structure of
CREPE to reduce processing and shorten the analysis window in order to reduce
latency.

### SASS-E: The Steelpan Audio Sample Set for Evaluation

The SASS-E dataset is an audio dataset curated as part of Colin's PhD research.
It consists of over 13,000 one-hit audio samples from three tenor steelpans
totaling over 9 hours and 25 minutes of audio. The samples were recorded in a
professional quality recording studio at 48 kHz/ 24-bit depth. Approximately 50
strikes were recorded per note per instrument at a wide variety of dynamic
levels and beating areas. This allows for comprehensive coverage of minute
details and fluctuations in timbre.

The audio samples are pre-split into training, validation, and test sets with
7,931 samples in the training set, 2,680 samples in the validation set, and
2,702 samples in the test set.

The audio files have filenames in the following format: <MIDI note
number>_<set>_<instrument label>_sample_<number>.wav. They are each labeled with
the MIDI note number for the given note struck. The instrument label is in the
format "ctenor-0x" where x is the number assigned to the instrument.

### Citation

Details for steelpan-pitch and SASS-E are available in a paper published in the
2023 NIME proceedings. We request that if you use this work or the SASS-E
dataset for your own project that you cite the following:

> [Steelpan-specific pitch detection: a dataset and deep learning model](https://nime.org/proceedings/2023/nime2023_59.pdf)<br>
> Colin Malloy, George Tzanetakis.<br>
> In Proceedings of the International Conference on New Interfaces for Musical Expression (NIME), 2023.

## Setup

Requires Python 3.11 and can be set up with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
```

TensorFlow 2.15 is used. On Apple Silicon the `tensorflow-macos` and
`tensorflow-metal` builds are selected automatically (and so is the legacy Adam
optimizer); other platforms use the standard `tensorflow` build.

## Usage

### 1. Download the dataset

```sh
uv run python src/download_sasse_dataset.py
```

This downloads the train, val, and test splits from Zenodo into `data/` and
extracts each into `data/<split>/`. Already-extracted splits are skipped, and a
pre-existing `SASSE_<split>.zip` is extracted without re-downloading.

```
usage: download_sasse_dataset [-h] [--data-dir DATA_DIR]
                              [--train | --no-train] [--val | --no-val]
                              [--test | --no-test]
                              [--keep-zips | --no-keep-zips]

options:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   Destination directory; split dirs and zips go under it
                        (created if missing) (default: data)
  --train, --no-train   Download the train split (default: on)
  --val, --no-val       Download the val split (default: on)
  --test, --no-test     Download the test split (default: on)
  --keep-zips, --no-keep-zips
                        Keep the zip files after extraction (default: on)
```

### 2. Train the model

```sh
uv run python src/steelpan_pitch_train.py
```

This expects audio in `data/train/` and `data/val/` (or under `--data-dir`),
trains up to 100 epochs with early stopping, and writes the best model to
`checkpoints/steelpan_pitch_best.keras` (monitoring `val_loss`,
`save_best_only`).

To continue training from an existing checkpoint (weights only — optimizer
state, epoch counter, and callback state are **not** restored):

```sh
uv run python src/steelpan_pitch_train.py --resume-from checkpoints/steelpan_pitch_best.keras
```

```
usage: steelpan_pitch_train [-h] [--step-size STEP_SIZE]
                            [--n-augment N_AUGMENT] [--batch-size BATCH_SIZE]
                            [--data-dir DATA_DIR]
                            [--checkpoints-dir CHECKPOINTS_DIR]
                            [--resume-from RESUME_FROM] [--seed SEED]
                            [--name NAME] [--epochs EPOCHS]
                            [--learning-rate LEARNING_RATE] [--verbose VERBOSE]
                            [--randomize-offset | --no-randomize-offset]

options:
  -h, --help            show this help message and exit
  --step-size STEP_SIZE
                        Samples between frame starts (default: 64)
  --n-augment N_AUGMENT
                        Pitch-shift augmentations per training recording;
                        validation is always 0 (default: 1)
  --batch-size BATCH_SIZE
                        Batch size (default: 128)
  --data-dir DATA_DIR   Base directory containing train/ and val/. If omitted,
                        use data/train and data/val
  --checkpoints-dir CHECKPOINTS_DIR
                        Where checkpoints are written (default: checkpoints)
  --resume-from RESUME_FROM
                        Path to a .keras file; weights-only load into a fresh
                        model
  --seed SEED           If given, seed random, numpy, and tensorflow; if
                        omitted, no seeding
  --name NAME           Keras model name (default: steelpan_pitch)
  --epochs EPOCHS       Max epochs; early stopping usually ends it sooner
                        (default: 100)
  --learning-rate LEARNING_RATE
                        Adam learning rate (default: 0.0002)
  --verbose VERBOSE     Passed through to model.fit(verbose=...) (default: 1)
  --randomize-offset, --no-randomize-offset
                        Random offset in [0, step_size) per file to
                        decorrelate framing (default: on)
```

## Data and checkpoints

- **Dataset** — `data/train/`, `data/val/`, `data/test/` (wav files, one
  directory per split), plus the source zips if kept. All of `data/` is
  gitignored; the download script is the canonical way to (re)create it.
- **Checkpoints** — written to `checkpoints/` (gitignored) as
  `<name>_best.keras` in the modern Keras format.


## References

[1] J. W. Kim, J. Salamon, P. Li, and J. P. Bello, "Crepe: A convolutional
representation for pitch estimation", *2018 IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP)*. Apr. 2018.

## License

Released under the MIT License; see [LICENSE](LICENSE).

