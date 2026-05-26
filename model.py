import tensorflow as tf
from config import *


def build_bird_classifier(num_classes=NUM_CLASSES):
    """Builds MobileNetV3 with Transfer Learning."""
    # Load pre-trained MobileNetV3 Large without top classification layers
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model
    base_model.trainable = False

    # Add new classification head
    model = tf.keras.Sequential([
        tf.keras.Input(shape=IMG_SIZE + (3,)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),  # Regularization
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    """
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)  # Regularization
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


def unfreeze_and_finetune(model, base_model, fine_tune_at=100):
    """Unfreezes top layers of base model for fine-tuning."""
    base_model.trainable = True

    # Freeze all layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Recompile with a lower learning rate for fine tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model