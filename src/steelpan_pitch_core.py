"""Shared, non-training functionality for steelpan-pitch.

This module is the single source of truth for the parts of the project that
are not training-specific:

- Fixed model/data constants (matching the research paper)
- The CREPE-style pitch classifier model builder
- The training data pipeline (frame slicing, augmentation, ``tf.data``)
- The prediction primitives (how the model's 360-bin output is read)

All functions here are ported from ``steelpan-crepe-test.ipynb``.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Suppress TensorFlow logging output (must happen before `import tensorflow`).
# Suppression levels: 0 = show all, 1 = hide INFO, 2 = hide INFO + WARNING,
# 3 = hide INFO + WARNING + ERROR. Respects a user-set value if present.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import librosa
import numpy as np
from numpy.lib.stride_tricks import as_strided  # noqa: F401  (used by pipeline impls)
import tensorflow as tf
from tensorflow.keras import layers

__all__ = [
    # Constants
    "WINDOW_SIZE",
    "MODEL_SRATE",
    "TOP_DB",
    # Model
    "build_model",
    # Training data pipeline
    "Recording",
    "load_audio_file",
    "get_shifted_audio",
    "process_audio",
    "process_recording",
    "recording_generator",
    "midi_to_targets",
    "make_dataset",
    # Prediction primitives
    "get_activation",
    "to_local_average_cents",
    "get_prediction",
]

# ---------------------------------------------------------------------------
# Constants (fixed to match the research paper; not user-configurable)
# ---------------------------------------------------------------------------

#: Number of samples per analysis frame.
WINDOW_SIZE: int = 128

#: Sample rate the model expects audio at.
MODEL_SRATE: int = 16000

#: Trim threshold (dB below peak) used when trimming recordings.
TOP_DB: float = 30.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(
    learning_rate: float,
    metrics: list[str] | None = None,
    name: str = "steelpan_pitch",
) -> tf.keras.Model:
    """Build and compile the CREPE-style CNN pitch classifier.

    The model classifies a single audio analysis frame over 360 pitch bins
    (5 cent each, covering MIDI notes 24--103).

    Architecture: ``Input(WINDOW_SIZE)`` -> ``Reshape(WINDOW_SIZE, 1, 1)`` ->
    6 x [Conv2D -> BatchNormalization -> MaxPool2D(2, 1) -> Dropout(0.25)]
    (filters ``[512, 64, 64, 64, 128, 256]``, kernel widths
    ``[64, 16, 16, 8, 4, 4]``) -> ``Permute((2, 1, 3))`` -> ``Flatten`` ->
    ``Dense(360, sigmoid)``.

    Note: on Apple Silicon (Darwin/arm64) the legacy Adam optimizer is used,
    as required by ``tensorflow-metal``; otherwise the standard Adam
    optimizer is used.

    Parameters
    ----------
    learning_rate : float
        Learning rate for the Adam optimizer.
    metrics : list[str], optional
        Metrics to record during training (e.g. ``["Accuracy"]``).
        If None, no metrics are recorded.
    name : str
        The Keras model name.

    Returns
    -------
    tf.keras.Model
        A newly initialized and compiled model.
    """
    filters = [512, 64, 64, 64, 128, 256]
    widths = [64, 16, 16, 8, 4, 4]

    x = layers.Input(shape=(WINDOW_SIZE,), name="input", dtype="float32")
    y = layers.Reshape(target_shape=(WINDOW_SIZE, 1, 1), name="input-reshape")(x)

    for i, (f, w) in enumerate(zip(filters, widths), start=1):
        y = layers.Conv2D(f, (w, 1), padding="same", activation="relu",
                          name=f"conv{i}")(y)
        y = layers.BatchNormalization(name=f"conv{i}-BN")(y)
        y = layers.MaxPool2D(pool_size=(2, 1), padding="valid",
                             name=f"conv{i}-maxpool")(y)
        y = layers.Dropout(0.25, name=f"conv{i}-dropout")(y)

    y = layers.Permute((2, 1, 3), name="transpose")(y)
    y = layers.Flatten(name="flatten")(y)
    y = layers.Dense(360, activation="sigmoid", name="classifier")(y)

    model = tf.keras.Model(inputs=x, outputs=y, name=name)

    # tensorflow-metal requires the legacy optimizer on Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        optimizer = tf.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=metrics,
    )

    return model


# ---------------------------------------------------------------------------
# Training data pipeline
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Recording:
    """All training data extracted from a single audio recording."""

    frames: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.float32]]:
        return iter(zip(self.frames, self.labels))


def load_audio_file(filepath: Path, offset: int = 0) -> tuple[np.ndarray, float]:
    """Load an audio file and extract its label.

    The audio is loaded (and resampled if necessary) at ``MODEL_SRATE`` as
    mono float32, and ``offset`` samples are dropped from the start.

    Parameters
    ----------
    filepath : Path
        Path to the audio file.
    offset : int, optional
        Number of samples to drop from the start of the audio. The training
        pipeline passes a random value in ``[0, step_size)`` so that framing
        does not line up identically across files; the default of 0 applies
        no offset.

    Returns
    -------
    audio : np.ndarray [shape=(num_samples,), dtype=np.float32]
        Raw unsliced audio data.
    label : float
        The target MIDI note number, parsed from the first underscore-delimited
        field of the filename (e.g. ``60_train_sample_1.wav`` -> 60.0).
    """
    audio, _ = librosa.load(filepath, sr=MODEL_SRATE, mono=True, dtype=np.float32)
    audio = audio[offset:]

    # The label is the MIDI note number encoded in the filename
    filename = os.path.basename(filepath).replace(".wav", "")
    label = float(filename.split("_")[0])

    return audio, label


def get_shifted_audio(audio: np.ndarray, label: float) -> tuple[np.ndarray, float]:
    """Pitch shift audio randomly by +/- 2 semitones in 20 cent increments.

    The shift amount is one of +/-0.2, +/-0.4, ..., +/-2.0 semitones, chosen
    uniformly at random (never zero). The returned label is shifted by the
    same amount so that it continues to describe the audio.

    Parameters
    ----------
    audio : np.ndarray
        Audio samples at ``MODEL_SRATE``.
    label : float
        The target MIDI note number for the audio.

    Returns
    -------
    audio_shifted : np.ndarray
        The pitch-shifted audio.
    label_shifted : float
        The target MIDI note number, shifted to match the audio.
    """
    # Random shift: a nonzero count of 20-cent steps, symmetric over
    # +/- 2 semitones (distance is 1..10, sign is +/-1).
    distance = np.random.randint(1, 11)
    sign = np.random.choice([-1, 1])
    n_steps = sign * distance * 0.2

    audio_shifted = librosa.effects.pitch_shift(audio, sr=MODEL_SRATE, n_steps=n_steps)
    label_shifted = label + n_steps

    return audio_shifted, label_shifted


def process_audio(
    audio: np.ndarray,
    window_size: int,
    step_size: int,
    label: float,
) -> Recording:
    """Slice audio into analysis frames and format the labels.

    Trims silence from the audio (``TOP_DB`` threshold), then uses
    ``as_strided`` to create overlapping frames of ``window_size`` samples
    spaced ``step_size`` apart. Each frame is normalized to zero mean and
    unit standard deviation. A uniform label array is created so every
    frame maps to the same MIDI note.

    Parameters
    ----------
    audio : np.ndarray [shape=(num_samples,)]
        Audio samples at ``MODEL_SRATE``.
    window_size : int
        The number of samples per analysis frame.
    step_size : int
        The number of samples between the starts of analysis frames.
    label : float
        Target MIDI note number for the audio.

    Returns
    -------
    Recording
        Framed, per-frame-normalized audio with uniform labels.
    """
    # Trim leading/trailing silence
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=TOP_DB)

    # Slice audio into overlapping frames via strided views
    n_frames = 1 + (len(audio_trimmed) - window_size) // step_size
    frames = as_strided(
        audio_trimmed,
        shape=(window_size, n_frames),
        strides=(audio_trimmed.itemsize, step_size * audio_trimmed.itemsize),
    )
    frames = frames.T.copy()  # shape: (n_frames, window_size), contiguous

    # Normalize each frame: zero mean, unit variance
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.std(frames, axis=1)[:, np.newaxis]

    # Every frame maps to the same MIDI note
    labels = np.full(frames.shape[0], label, dtype=np.float32)

    return Recording(frames=frames, labels=labels)


def process_recording(
    filepath: Path,
    window_size: int,
    step_size: int,
    n_augment: int = 0,
    randomize_offset: bool = True,
) -> Iterator[Recording]:
    """Process one WAV file into its original and augmented recordings.

    Loads the audio with a random offset in ``[0, step_size)`` (via
    ``load_audio_file``) so that framing does not line up identically across
    files, then slices and augments it.

    Parameters
    ----------
    filepath : Path
        Path to the audio file.
    window_size : int
        The number of samples per analysis frame.
    step_size : int
        The number of samples between the starts of analysis frames.
    n_augment : int
        The number of pitch-shifted augmentations to produce.
    randomize_offset : bool
        If True (default), load the audio with a random offset in
        ``[0, step_size)`` to decorrelate framing across files. If False,
        no offset is applied.

    Yields
    ------
    Recording
        The original recording followed by ``n_augment`` pitch-shifted ones.
    """
    if randomize_offset:
        offset = np.random.randint(0, step_size)
    else:
        offset = 0
    audio, label = load_audio_file(filepath, offset=offset)

    # Original recording
    yield process_audio(audio, window_size, step_size, label)

    # Pitch-shifted augmentations
    for _ in range(n_augment):
        audio_shifted, label_shifted = get_shifted_audio(audio, label)
        yield process_audio(audio_shifted, window_size, step_size, label_shifted)


def recording_generator(
    data_dir: Path,
    window_size: int,
    step_size: int,
    n_augment: int = 0,
    randomize_offset: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Lazily process all audio files in a directory in shuffled order.

    Each file produces one original Recording plus ``n_augment`` pitch-shifted
    Recordings. Files are processed in a randomly shuffled order.

    Parameters
    ----------
    data_dir : Path
        Directory containing the ``.wav`` files to process.
    window_size : int
        The number of samples per analysis frame.
    step_size : int
        The number of samples between the starts of analysis frames.
    n_augment : int
        The number of pitch-shifted augmentations per recording.
    randomize_offset : bool
        Passed through to ``process_recording``.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        ``(frames, labels)`` pairs, one per (possibly augmented) recording.
    """
    wav_paths = [p for p in data_dir.iterdir() if p.suffix == '.wav']
    np.random.shuffle(wav_paths)

    for path in wav_paths:
        for recording in process_recording(
            filepath=path,
            window_size=window_size,
            step_size=step_size,
            n_augment=n_augment,
            randomize_offset=randomize_offset,
        ):
            yield recording.frames, recording.labels


def midi_to_targets(labels: tf.Tensor) -> tf.Tensor:
    """Convert MIDI labels to CREPE-style 360-bin targets.

    Each label is mapped to a Gaussian-softened 360-bin vector. Bin 0
    corresponds to MIDI 24 (0 cents), bin 1 to 5 cents, …, bin 359 to
    MIDI 103 (7175 cents). The Gaussian is centered on the true pitch with
    σ = 25 cents.

    Parameters
    ----------
    labels : tf.Tensor [shape=(B,)]
        MIDI note numbers.

    Returns
    -------
    tf.Tensor [shape=(B, 360)]
        Soft targets for the 360 pitch bins.
    """
    cents_true = 5.0 * (labels - 24.0)
    cents_i = tf.range(360, dtype=tf.float32)

    cents_diff = cents_i[tf.newaxis, :] - cents_true[:, tf.newaxis]
    return tf.exp(-tf.square(20.0 * cents_diff) / (2.0 * 25.0**2))


def make_dataset(
    data_dir: Path,
    window_size: int,
    step_size: int,
    n_augment: int = 0,
    shuffle_buffer_size: int = 10_000,
    batch_size: int = 128,
    randomize_offset: bool = True,
) -> tf.data.Dataset:
    """Create a ``tf.data.Dataset`` that lazily processes audio files.

    The pipeline:

    1. ``from_generator`` wraps ``recording_generator`` to yield
       ``(frames, labels)`` pairs.
    2. ``flat_map`` expands each recording into individual
       ``(frame, label)`` pairs.
    3. ``shuffle`` randomizes the frame-level order each iteration.
    4. ``batch`` groups frames into batches.
    5. ``map`` converts MIDI labels to Gaussian-softened 360-bin targets
       via ``midi_to_targets``.
    6. ``prefetch`` overlaps processing and model execution.

    Parameters
    ----------
    data_dir : Path
        Directory containing the ``.wav`` files to process.
    window_size : int
        The number of samples per analysis frame.
    step_size : int
        The number of samples between the starts of analysis frames.
    n_augment : int
        The number of pitch-shifted augmentations per recording.
    shuffle_buffer_size : int
        Size of the frame-level shuffle buffer.
    batch_size : int
        The number of frames per batch.
    randomize_offset : bool
        Passed through to ``recording_generator`` and ``process_recording``.

    Returns
    -------
    tf.data.Dataset
        A dataset of ``(frames [B, window_size], targets [B, 360])`` batches.
    """
    recording_dataset = tf.data.Dataset.from_generator(
        lambda: recording_generator(
            data_dir=data_dir,
            window_size=window_size,
            step_size=step_size,
            n_augment=n_augment,
            randomize_offset=randomize_offset,
        ),
        output_signature=(
            tf.TensorSpec(
                shape=(None, window_size),
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(None,),
                dtype=tf.float32,
            ),
        ),
    )

    # Convert each recording into individual (frame, label) pairs
    frame_dataset = recording_dataset.flat_map(
        lambda frames, labels: tf.data.Dataset.from_tensor_slices(
            (frames, labels)
        )
    )

    return (
        frame_dataset
        .shuffle(
            buffer_size=shuffle_buffer_size,
            reshuffle_each_iteration=True,
        )
        .batch(batch_size=batch_size)
        .map(
            lambda frames, labels: (frames, midi_to_targets(labels)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )


# ---------------------------------------------------------------------------
# Prediction primitives
# ---------------------------------------------------------------------------


def get_activation(
    audio: np.ndarray,
    model: tf.keras.Model,
    window_size: int,
    step_size: int,
    center: bool = True,
    verbose: int = 1,
) -> np.ndarray:
    """Frame the audio and run the model to get the activation matrix.

    If ``center`` is True, the audio is zero-padded by ``window_size // 2``
    samples on each side so that frames are centered around their timestamps.
    Frames are created via ``as_strided``, normalized to zero mean and unit
    variance, then passed through the model.

    Parameters
    ----------
    audio : np.ndarray
        Audio samples at ``MODEL_SRATE``.
    model : tf.keras.Model
        A compiled model to run prediction with.
    window_size : int
        The number of samples per analysis frame (must match training).
    step_size : int
        The number of samples between the starts of analysis frames.
    center : bool
        If True, pad the audio so frames are centered around their timestamps.
    verbose : int
        Verbosity level for ``model.predict``.

    Returns
    -------
    np.ndarray [shape=(T, 360)]
        The raw activation matrix for each frame, where ``T`` is the number
        of frames processed.
    """
    # Pad so that frames are centered around their timestamps
    if center:
        audio = np.pad(audio, window_size // 2, mode='constant', constant_values=0)

    # Make `window_size`-sample frames of the audio
    n_frames = 1 + (len(audio) - window_size) // step_size
    frames = as_strided(
        audio,
        shape=(window_size, n_frames),
        strides=(audio.itemsize, step_size * audio.itemsize),
    )
    frames = frames.T.copy()

    # Normalize each frame
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.std(frames, axis=1)[:, np.newaxis]

    return model.predict(frames, verbose=verbose)


# Cached bin-number-to-cents mapping for to_local_average_cents.
# Bin 0 = 2051.32 cents (≈ MIDI 24), bin 359 = 9231.32 cents (≈ MIDI 103).
# This is the CREPE model's standard mapping.
_CENTS_MAPPING = np.linspace(0, 7180, 360) + 2051.3179423647566


def to_local_average_cents(
    salience: np.ndarray,
    center: int | None = None,
) -> np.ndarray | float:
    """Find the weighted average cents near the argmax bin.

    Uses a local window of 10 bins (center-4 … center+4) around the peak
    to compute a weighted average of the bin cents values. This refines the
    argmax into a sub-bin estimate.

    Parameters
    ----------
    salience : np.ndarray
        A 1-D or 2-D array of bin activations.
    center : int, optional
        The bin to average around. If None, the argmax bin is used.

    Returns
    -------
    np.ndarray | float
        The local average cents value (per row for 2-D input).
    """
    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * _CENTS_MAPPING[start:end]
        )
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return np.array(
            [to_local_average_cents(salience[i, :]) for i in range(salience.shape[0])]
        )

    raise ValueError("salience should be a 1-D or 2-D ndarray")


def get_prediction(
    audio: np.ndarray,
    model: tf.keras.Model,
    window_size: int,
    step_size: int,
    time_offset: float = 0.0,
    verbose: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get time, frequency, and activation predictions for audio.

    Runs the model on the audio to produce a salience matrix, then converts
    it to local-average cents and finally to Hz. The time axis is computed
    as evenly-spaced frame timestamps, with an optional ``time_offset`` so
    callers can account for leading trim.

    Parameters
    ----------
    audio : np.ndarray
        Audio samples at ``MODEL_SRATE``.
    model : tf.keras.Model
        A compiled model to run prediction with.
    window_size : int
        The number of samples per analysis frame (must match training).
    step_size : int
        The number of samples between the starts of analysis frames.
    time_offset : float
        Seconds to add to every frame time (e.g. the amount of leading trim).
    verbose : int
        Verbosity level for ``model.predict``.

    Returns
    -------
    time : np.ndarray [shape=(T,)]
        Timestamps (seconds) for the predictions.
    frequency : np.ndarray [shape=(T,)]
        Frequency (Hz) predictions; 0 where the prediction is undefined.
    activation : np.ndarray [shape=(T, 360)]
        The raw activation matrix for each frame.
    """
    activation = get_activation(
        audio, model, window_size, step_size, verbose=verbose
    )

    cents = to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    frequency[np.isnan(frequency)] = 0

    # Evenly spaced frame timestamps, accounting for center padding and trim
    time = np.arange(activation.shape[0]) * step_size / MODEL_SRATE + time_offset

    return time, frequency, activation
