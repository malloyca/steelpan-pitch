"""Training script for the steelpan-pitch model.

Parses CLI arguments, validates the data directories, and runs the training
loop. The model builder, constants, and data pipeline are imported from
``steelpan_pitch_core``.

Run from the repo root::

    uv run python src/steelpan_pitch_train.py [options]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from steelpan_pitch_core import WINDOW_SIZE, build_model, make_dataset
import tensorflow as tf
from tqdm import tqdm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str], optional
        Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The parsed arguments (dashes converted to underscores).
    """
    parser = argparse.ArgumentParser(
        prog="steelpan_pitch_train",
        description="Train the CREPE-style steelpan pitch classifier.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=WINDOW_SIZE // 2,
        help="Samples between frame starts (default: %(default)s)",
    )
    parser.add_argument(
        "--n-augment",
        type=int,
        default=1,
        help=(
            "Pitch-shift augmentations per training recording; "
            "validation is always 0 (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Base directory containing train/ and val/. "
            "If omitted, use data/train and data/val"
        ),
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Where checkpoints are written (default: %(default)s)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to a .keras file; weights-only load into a fresh model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If given, seed random, numpy, and tensorflow; if omitted, no seeding",
    )
    parser.add_argument(
        "--name",
        default="steelpan_pitch",
        help="Keras model name (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max epochs; early stopping usually ends it sooner (default: %(default)s)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Adam learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Passed through to model.fit(verbose=...) (default: %(default)s)",
    )
    parser.add_argument(
        "--randomize-offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Random offset in [0, step_size) per file to decorrelate framing (default: on)",
    )
    return parser.parse_args(argv)


def resolve_data_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve the train and validation data directories from parsed args.

    If ``--data-dir`` was given, the effective directories are
    ``<data-dir>/train`` and ``<data-dir>/val``; otherwise the defaults
    ``data/train`` and ``data/val`` are used.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (from :func:`parse_args`).

    Returns
    -------
    tuple[Path, Path]
        ``(train_dir, val_dir)``.
    """
    if args.data_dir is not None:
        return args.data_dir / "train", args.data_dir / "val"
    return Path("data") / "train", Path("data") / "val"


def validate_data_dirs(train_dir: Path, val_dir: Path) -> None:
    """Validate that the data directories exist and contain audio.

    Each directory must exist and contain at least one ``.wav`` file.

    Parameters
    ----------
    train_dir : Path
        The training data directory.
    val_dir : Path
        The validation data directory.

    Raises
    ------
    FileNotFoundError
        If a data directory does not exist.
    ValueError
        If a data directory does not contain at least one ``.wav`` file.
    """
    for d in (train_dir, val_dir):
        if not d.is_dir():
            raise FileNotFoundError(
                f"Data directory not found: {d}. "
                "Run `python src/download_sasse_dataset.py` first to download the dataset."
            )
        if not any(p.suffix == ".wav" for p in d.iterdir()):
            raise ValueError(
                f"No .wav files found in {d}. "
                "Run `python src/download_sasse_dataset.py` first to download the dataset."
            )


def set_seed(seed: int) -> None:
    """Seed the random number generators for reproducibility.

    Seeds Python's ``random`` module, NumPy's global RNG, and
    TensorFlow's RNG.

    Parameters
    ----------
    seed : int
        The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def print_gpu_info() -> None:
    """Print the detected GPU devices (or note their absence).

    Informational only, kept from the research notebook. The device is not
    otherwise configured; Keras/Metal selects it automatically.
    """
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        print("GPUs found:")
        for gpu in gpu_devices:
            print(f"- {gpu}")
    else:
        print("No GPU found")


def setup_model(
    learning_rate: float,
    name: str,
    resume_from: Path | None = None,
) -> tf.keras.Model:
    """Build the model and optionally load weights to resume training.

    Parameters
    ----------
    learning_rate : float
        Learning rate for the Adam optimizer.
    name : str
        Keras model name.
    resume_from : Path, optional
        Path to a ``.keras`` file whose weights are loaded into the fresh
        model (weights-only semantics: optimizer state, epoch counter, and
        callback state are not restored).

    Returns
    -------
    tf.keras.Model
        A compiled, freshly initialized model (with loaded weights if
        ``resume_from`` was given).
    """
    model = build_model(
        learning_rate=learning_rate,
        metrics=["Accuracy"],
        name=name,
    )
    if resume_from is not None:
        model.load_weights(resume_from)
        print(f"Resuming from weights: {resume_from}")
    return model


def count_dataset_steps(dataset: tf.data.Dataset, desc: str = "steps") -> int:
    """Count the number of batches in a dataset by iterating it once.

    Iterating also runs the full pipeline once (audio loading, framing,
    augmentation), which doubles as a smoke test. A fresh pass over the
    dataset is started on each iteration, so the dataset remains usable
    afterwards (e.g. for ``.repeat()`` in ``model.fit``).

    Parameters
    ----------
    dataset : tf.data.Dataset
        The (non-repeated) dataset to count.
    desc : str
        Description for the ``tqdm`` progress bar.

    Returns
    -------
    int
        The number of batches in one full pass over the dataset.
    """
    return sum(1 for _ in tqdm(dataset, desc=desc))


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, validate the data, and run the training loop.

    Parameters
    ----------
    argv : list[str], optional
        Argument list to parse. Defaults to ``sys.argv[1:]``.

    Raises
    ------
    FileNotFoundError
        If a data directory does not exist.
    ValueError
        If a data directory does not contain at least one ``.wav`` file.
    """
    args = parse_args(argv)

    train_dir, val_dir = resolve_data_dirs(args)
    validate_data_dirs(train_dir, val_dir)

    if args.seed is not None:
        set_seed(args.seed)

    print_gpu_info()

    model = setup_model(
        learning_rate=args.learning_rate,
        name=args.name,
        resume_from=args.resume_from,
    )

    train_dataset = make_dataset(
        data_dir=train_dir,
        window_size=WINDOW_SIZE,
        step_size=args.step_size,
        n_augment=args.n_augment,
        shuffle_buffer_size=50_000,
        batch_size=args.batch_size,
        randomize_offset=args.randomize_offset,
    )
    val_dataset = make_dataset(
        data_dir=val_dir,
        window_size=WINDOW_SIZE,
        step_size=args.step_size,
        n_augment=0,
        shuffle_buffer_size=50_000,
        batch_size=args.batch_size,
        randomize_offset=args.randomize_offset,
    )

    train_steps = count_dataset_steps(train_dataset, "Training steps")
    val_steps = count_dataset_steps(val_dataset, "Validation steps")

    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(args.checkpoints_dir / f"{args.name}_best.keras"),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        restore_best_weights=True,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
    )
    callbacks = [
        checkpoint,
        early_stopping,
        reduce_lr,
    ]

    model.fit(
        train_dataset.repeat(),
        validation_data=val_dataset.repeat(),
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
