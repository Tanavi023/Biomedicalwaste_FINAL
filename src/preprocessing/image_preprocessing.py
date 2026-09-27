"""TensorFlow image transforms shared by training and inference pipelines."""

from __future__ import annotations

from src.utils.runtime_environment import configure_runtime_environment

configure_runtime_environment()

import tensorflow as tf

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
RANDOM_SEED = 42


def build_augmentation_layer(seed: int = RANDOM_SEED) -> tf.keras.Sequential:
    """Return label-preserving augmentation applied only to training images.

    Colour-hue transforms are intentionally excluded: disposal categories must
    not be distorted by aggressive colour changes. The small geometric and
    contrast variations simulate realistic camera position and lighting changes.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(mode="horizontal", seed=seed),
            tf.keras.layers.RandomRotation(
                factor=0.08,
                fill_mode="reflect",
                interpolation="bilinear",
                seed=seed + 1,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.10, 0.10),
                width_factor=(-0.10, 0.10),
                fill_mode="reflect",
                interpolation="bilinear",
                seed=seed + 2,
            ),
            tf.keras.layers.RandomContrast(factor=0.10, seed=seed + 3),
        ],
        name="training_augmentation",
    )


def resize_image(image: tf.Tensor) -> tf.Tensor:
    """Resize an RGB image to the MobileNetV2 input resolution as float32."""
    image = tf.cast(image, tf.float32)
    return tf.image.resize(image, size=IMAGE_SIZE, method="bilinear", antialias=True)


def normalize_for_mobilenet_v2(image: tf.Tensor) -> tf.Tensor:
    """Scale an image with 0-255 pixels to MobileNetV2's [-1, 1] range."""
    image = tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 255.0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)


def preprocess_for_inference(image: tf.Tensor) -> tf.Tensor:
    """Resize and normalize an image without any random transformation."""
    return normalize_for_mobilenet_v2(resize_image(image))


def preprocess_for_training(image: tf.Tensor, augmentation_layer: tf.keras.layers.Layer) -> tf.Tensor:
    """Resize, augment, and normalize one image for a training-only pathway."""
    resized_image = resize_image(image)
    augmented_image = augmentation_layer(resized_image, training=True)
    return normalize_for_mobilenet_v2(augmented_image)


def denormalize_for_display(image: tf.Tensor) -> tf.Tensor:
    """Convert MobileNetV2-normalized pixels back to displayable [0, 1] RGB."""
    return tf.clip_by_value((tf.cast(image, tf.float32) + 1.0) / 2.0, 0.0, 1.0)
