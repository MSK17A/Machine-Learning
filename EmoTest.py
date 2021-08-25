from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kt_utils import *

model = keras.models.load_model("EmoNet")

img = mpimg.imread("Testing Images\Smile1.jpg")
img = np.expand_dims(img, axis=0)
imgR = tf.image.resize_with_pad(img, 64, 64) / 255

plt.imshow(imgR[0,:,:,:])
print(np.argmax(model.predict(imgR)))
plt.show()