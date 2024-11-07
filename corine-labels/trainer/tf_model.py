# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This trains a TensorFlow Keras model to classify land cover.

The model is a simple Fully Convolutional Network (FCN).
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



# Default values.
EPOCHS = 100
BATCH_SIZE = 512
KERNEL_SIZE = 5
TRAIN_TEST_RATIO = 90  # percent for training, the rest for testing/validation
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 8
EARLY_STOPPING_PATIENCE = 1000

# Constants.
CORINE_CLASS_NAMES = [
    "Artificial surfaces; urban fabric; continuous urban fabric",
    "Artificial surfaces; urban fabric; discontinuous urban fabric",
    "Artificial surfaces; industrial, commercial, and transport units; industrial or commercial units",
    "Artificial surfaces; industrial, commercial, and transport units; road and rail networks and associated land",
    "Artificial surfaces; industrial, commercial, and transport units; port areas",
    "Artificial surfaces; industrial, commercial, and transport units; airports",
    "Artificial surfaces; mine, dump, and construction sites; mineral extraction sites",
    "Artificial surfaces; mine, dump, and construction sites; dump sites",
    "Artificial surfaces; mine, dump, and construction sites; construction sites",
    "Artificial surfaces; artificial, non-agricultural vegetated areas; green urban areas",
    "Artificial surfaces; artificial, non-agricultural vegetated areas; sport and leisure facilities",
    "Agricultural areas; arable land; non-irrigated arable land",
    "Agricultural areas; arable land; permanently irrigated land",
    "Agricultural areas; arable land; rice fields",
    "Agricultural areas; permanent crops; vineyards",
    "Agricultural areas; permanent crops; fruit trees and berry plantations",
    "Agricultural areas; permanent crops; olive groves",
    "Agricultural areas; pastures; pastures",
    "Agricultural areas; heterogeneous agricultural areas; annual crops associated with permanent crops",
    "Agricultural areas; heterogeneous agricultural areas; complex cultivation patterns",
    "Agricultural areas; heterogeneous agricultural areas; land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agricultural areas; heterogeneous agricultural areas; agro-forestry areas",
    "Forest and semi natural areas; forests; broad-leaved forest",
    "Forest and semi natural areas; forests; coniferous forest",
    "Forest and semi natural areas; forests; mixed forest",
    "Forest and semi natural areas; scrub and/or herbaceous vegetation associations; natural grasslands",
    "Forest and semi natural areas; scrub and/or herbaceous vegetation associations; moors and heathland",
    "Forest and semi natural areas; scrub and/or herbaceous vegetation associations; sclerophyllous vegetation",
    "Forest and semi natural areas; scrub and/or herbaceous vegetation associations; transitional woodland-shrub",
    "Forest and semi natural areas; open spaces with little or no vegetation; beaches, dunes, sands",
    "Forest and semi natural areas; open spaces with little or no vegetation; bare rocks",
    "Forest and semi natural areas; open spaces with little or no vegetation; sparsely vegetated areas",
    "Forest and semi natural areas; open spaces with little or no vegetation; burnt areas",
    "Forest and semi natural areas; open spaces with little or no vegetation; glaciers and perpetual snow",
    "Wetlands; inland wetlands; inland marshes",
    "Wetlands; inland wetlands; peat bogs",
    "Wetlands; maritime wetlands; salt marshes",
    "Wetlands; maritime wetlands; salines",
    "Wetlands; maritime wetlands; intertidal flats",
    "Water bodies; inland waters; water courses",
    "Water bodies; inland waters; water bodies",
    "Water bodies; marine waters; coastal lagoons",
    "Water bodies; marine waters; estuaries",
    "Water bodies; marine waters; sea and ocean"
]
CORINE_CLASS_VALUES = [
    111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142, 211, 212, 213, 221, 
    222, 223, 231, 241, 242, 243, 244, 311, 312, 313, 321, 322, 323, 324, 331, 
    332, 333, 334, 335, 411, 412, 421, 422, 423, 511, 512, 521, 522, 523]
NUM_CLASSES = len(CORINE_CLASS_VALUES)
NUM_INPUTS = 13

def read_example(serialized: bytes) -> tuple[tf.Tensor, tf.Tensor]:
    """Parses and reads a training example from TFRecords.

    Args:
        serialized: Serialized example bytes from TFRecord files.

    Returns: An (inputs, labels) pair of tensors.
    """
    features_dict = {
        "inputs": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(serialized, features_dict)
    inputs = tf.io.parse_tensor(example["inputs"], tf.float32)
    labels = tf.io.parse_tensor(example["labels"], tf.uint8)

    # TensorFlow cannot infer the shape's rank, so we set the shapes explicitly.
    inputs.set_shape([None, None, NUM_INPUTS])
    labels.set_shape([None, None, 1])

    # Classifications are measured against one-hot encoded vectors.
    one_hot_labels = tf.one_hot(labels[:, :, 0], NUM_CLASSES)
    return (inputs, one_hot_labels)


def read_dataset(data_path: str) -> tf.data.Dataset:
    """Reads compressed TFRecord files from a directory into a tf.data.Dataset.

    Args:
        data_path: Local or Cloud Storage directory path where the TFRecord files are.

    Returns: A tf.data.Dataset with the contents of the TFRecord files.
    """
    file_pattern = tf.io.gfile.join(data_path, "*.tfrecord.gz")
    file_names = tf.data.Dataset.list_files(file_pattern).cache()
    dataset = tf.data.TFRecordDataset(file_names, compression_type="GZIP")
    return dataset.map(read_example, num_parallel_calls=tf.data.AUTOTUNE)


def split_dataset(
    dataset: tf.data.Dataset,
    batch_size: int = BATCH_SIZE,
    train_test_ratio: int = TRAIN_TEST_RATIO,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Splits a dataset into training and validation subsets.

    Args:
        dataset: Full dataset with all the training examples.
        batch_size: Number of examples per training batch.
        train_test_ratio: Percent of the data to use for training.

    Returns: A (training, validation) dataset pair.
    """
    # For more information on how to optimize your tf.data.Dataset, see:
    #   https://www.tensorflow.org/guide/data_performance
    indexed_dataset = dataset.enumerate()  # add an index to each example
    train_dataset = (
        indexed_dataset.filter(lambda i, _: i % 100 <= train_test_ratio)
        .map(lambda _, data: data, num_parallel_calls=tf.data.AUTOTUNE)  # remove index
        .cache()  # cache the individual parsed examples
        .shuffle(SHUFFLE_BUFFER_SIZE)  # randomize the examples for the batches
        .batch(batch_size)  # batch randomized examples
        .prefetch(tf.data.AUTOTUNE)  # prefetch the next batch
    )
    validation_dataset = (
        indexed_dataset.filter(lambda i, _: i % 100 > train_test_ratio)
        .map(lambda _, data: data, num_parallel_calls=tf.data.AUTOTUNE)  # remove index
        .batch(batch_size)  # batch the parsed examples, no need to shuffle
        .cache()  # cache the batches of examples
        .prefetch(tf.data.AUTOTUNE)  # prefetch the next batch
    )
    return (train_dataset, validation_dataset)


def create_model(
    dataset: tf.data.Dataset, kernel_size: int = KERNEL_SIZE
) -> tf.keras.Model:
    """Creates a Fully Convolutional Network Keras model.

    Make sure you pass the *training* dataset, not the validation or full dataset.

    Args:
        dataset: Training dataset used to normalize inputs.
        kernel_size: Size of the square of neighboring pixels for the model to look at.

    Returns: A compiled fresh new model (not trained).
    """
    # Adapt the preprocessing layers.
    normalization = tf.keras.layers.Normalization()
    normalization.adapt(dataset.map(lambda inputs, _: inputs))

    # Define the Fully Convolutional Network.
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(None, None, NUM_INPUTS)),
            normalization,
            tf.keras.layers.Conv2D(32, kernel_size, activation="relu"),
            tf.keras.layers.Conv2DTranspose(16, kernel_size, activation="relu"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.OneHotIoU(
                num_classes=NUM_CLASSES,
                target_class_ids=list(range(NUM_CLASSES)),
            ),
            'accuracy'
        ],
    )
    return model


def run(
    data_path: str,
    model_path: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    kernel_size: int = KERNEL_SIZE,
    train_test_ratio: int = TRAIN_TEST_RATIO,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
) -> tf.keras.Model:
    """Creates and trains the model.

    Args:
        data_path: Local or Cloud Storage directory path where the TFRecord files are.
        model_path: Local or Cloud Storage directory path to store the trained model.
        epochs: Number of times the model goes through the training dataset during training.
        batch_size: Number of examples per training batch.
        kernel_size: Size of the square of neighboring pixels for the model to look at.
        train_test_ratio: Percent of the data to use for training.
        early_stopping_patience: Early stopping patience.


    Returns: The trained model.
    """
    print(f"data_path: {data_path}")
    print(f"model_path: {model_path}")
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"kernel_size: {kernel_size}")
    print(f"train_test_ratio: {train_test_ratio}")
    print(f"early_stopping_patience: {early_stopping_patience}")
    print("-" * 40)

    dataset = read_dataset(data_path)
    (train_dataset, test_dataset) = split_dataset(dataset, batch_size, train_test_ratio)

    model = create_model(train_dataset, kernel_size)
    print(model.summary())

    # Create a Tensorboard callback and write to the gcs path provided by AIP_TENSORBOARD_LOG_DIR
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR'],
        histogram_freq=1)
    
    # Add early stopping callback.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=early_stopping_patience, 
        restore_best_weights=True)

    class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
        def __init__(self, validation_data, writer):
            super().__init__()
            self.validation_data = validation_data
            self.writer = writer

        def on_epoch_end(self, epoch, logs=None):
            cm = tf.zeros((NUM_CLASSES, NUM_CLASSES), dtype=tf.dtypes.int32)
            # 1. Get predictions and true labels
            for x, y in self.validation_data:
                for i in range(x.shape[0]):

                    label = tf.math.argmax(y, axis=-1)
                    label_np = np.array(label)
                    label_flat = tf.reshape(label_np, [-1]) 

                    print(f'shape of x: {x.shape}')
                    print(f'shape of xi: {x[i].shape}')

                    probabilities = self.model.predict(np.stack([x[i]]), verbose=0)[0]
                    predictions = probabilities.argmax(axis=-1)
                    predictions_flat = tf.reshape(predictions, [-1])

                # 2. Calculate confusion matrix (using tf.math.confusion_matrix)
                    batch_cm = tf.math.confusion_matrix(
                        label_flat, 
                        predictions_flat,
                        num_classes=NUM_CLASSES
                    )
                    cm += batch_cm

            # 3. Create a confusion matrix image
            figure = plt.figure(figsize=(8, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("Confusion matrix")
            plt.colorbar()

            # ... (add labels and formatting) ...

            # 4. Convert to image summary
            figure.canvas.draw()
            img = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
            img = np.expand_dims(img, 0)  # Add batch dimension

            # 5. Log to TensorBoard
            with self.writer.as_default():
                tf.summary.image("Confusion Matrix", img, step=epoch)
            plt.close(figure)

    # ... (set up TensorBoard and writer) ...
    log_dir = "logs/confusion_matrix/"
    writer = tf.summary.create_file_writer(log_dir)

    cm_callback = ConfusionMatrixCallback(validation_data=test_dataset, writer=writer)


    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=[tensorboard_callback, early_stopping, cm],
    )
    model.save(model_path)
    print(f"Model saved to path: {model_path}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        help="Local or Cloud Storage directory path where the TFRecord files are.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local or Cloud Storage directory path to store the trained model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of times the model goes through the training dataset during training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Number of examples per training batch.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=KERNEL_SIZE,
        help="Size of the square of neighboring pixels for the model to look at.",
    )
    parser.add_argument(
        "--train-test-ratio",
        type=int,
        default=TRAIN_TEST_RATIO,
        help="Percent of the data to use for training.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Early stopping patience.",
    )
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        kernel_size=args.kernel_size,
        train_test_ratio=args.train_test_ratio,
        early_stopping_patience=args.early_stopping_patience
    )
