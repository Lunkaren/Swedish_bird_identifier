import tensorflow as tf
import os
import json
from config import *
from dataset import prepare_dataset
from model import build_bird_classifier, unfreeze_and_finetune
import matplotlib.pyplot as plt


def train():
    create_label_mapping()

    # 1. Prepare Datasets
    train_dataset = prepare_dataset(DATA_DIR, subset="training")
    val_dataset = prepare_dataset(DATA_DIR, subset="validation")

    # 2. Build Model
    model, base_model = build_bird_classifier()

    # 3. Callbacks setup
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)


    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "best_model.keras"),
            save_best_only=True,
            monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1
        )
    ]

    # 4. Train Model (Feature Extraction)
    print("Starting feature extraction training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS // 2,
        callbacks=callbacks
    )

    # 5. Fine-tuning
    print("Starting fine-tuning...")
    model = unfreeze_and_finetune(model, base_model)

    history_finetune = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        initial_epoch=history.epoch[-1],
        callbacks=callbacks
    )

    plot_and_save_history(history, history_finetune)

    # 6. Save final production model
    save_dir = os.path.dirname(MODEL_SAVE_PATH)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

def create_label_mapping():
    class_names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

    # Save the labels list to a JSON file
    save_dir = os.path.dirname(MODEL_SAVE_PATH)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(LABELS_PATH, 'w') as f:
        json.dump(class_names, f, indent=4)
    print(f"Saved {len(class_names)} class labels to {LABELS_PATH}")


def plot_and_save_history(history, history_finetune, save_path="training_history.png"):
    # Combine the histories
    acc = history.history['accuracy'] + history_finetune.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
    loss = history.history['loss'] + history_finetune.history['loss']
    val_loss = history.history['val_loss'] + history_finetune.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # Mark the point where fine-tuning started
    plt.axvline(x=len(history.history['accuracy']) - 1, color='r', linestyle='--', label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.axvline(x=len(history.history['loss']) - 1, color='r', linestyle='--', label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig(save_path)
    print(f"Saved training graphs to {save_path}")

if __name__ == "__main__":
    train()