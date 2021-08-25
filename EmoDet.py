import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kt_utils import *

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()


def model(input_shape):

    inputs = keras.Input(shape=input_shape, name='img')
    a = layers.Conv2D(64, 3, activation="relu")(inputs)
    a = layers.ZeroPadding2D((3,3))(a)
    a = layers.Conv2D(64, 3, activation="relu")(a)
    a = layers.MaxPooling2D(2)(a)

    a = layers.Conv2D(128, 3, activation="relu")(a)
    a = layers.ZeroPadding2D((3,3))(a)
    a = layers.Conv2D(128, 3, activation="relu")(a)
    a = layers.MaxPooling2D(2)(a)

    a = layers.Conv2D(256, 3, activation="relu")(a)
    a = layers.ZeroPadding2D((3,3))(a)
    a = layers.Conv2D(256, 3, activation="relu")(a)
    a = layers.ZeroPadding2D((3,3))(a)
    a = layers.Conv2D(256, 3, activation="relu")(a)
    a = layers.MaxPooling2D(2)(a)

    a = layers.Conv2D(256, 3, activation="relu")(a)
    a = layers.ZeroPadding2D((3,3))(a)
    a = layers.Conv2D(256, 3, activation="relu")(a)
    a = layers.ZeroPadding2D((3,3))(a)
    a = layers.Conv2D(256, 3, activation="relu")(a)
    a = layers.MaxPooling2D(2)(a)

    a = layers.Flatten()(a)
    a = layers.Dense(4096)(a)
    a = layers.Dense(16)(a)

    model = keras.Model(inputs, a, name="Happy_House")
    model.summary()
    keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
    return model

Model = model((X_train_orig[0,:,:,:].shape))

Model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)

history = Model.fit(X_train_orig, Y_train_orig.T, batch_size=16, epochs=40, validation_split=0.2)

test_scores = Model.evaluate(X_test_orig, Y_test_orig.T, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

Model.save("EmoNet")

del Model