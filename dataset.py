import tensorflow as tf
from config import *


def prepare_dataset(data_dir, subset="training", validation_split=0.2):
    """Loads and prepares the tf.data.Dataset with optimizations."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset=subset,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    # On-the-fly Data Augmentation pipeline (Applied only to training data)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

    # Preprocessing specific to MobileNetV3
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    AUTOTUNE = tf.data.AUTOTUNE

    if subset == "training":
        # Apply data augmentation
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Normalize input
    dataset = dataset.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

    # Cache and prefetch to avoid CPU-GPU bottleneck
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset