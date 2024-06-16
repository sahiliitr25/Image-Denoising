
import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMAGE_SIZE = 256
BATCH_SIZE = 16
MAX_TRAIN_IMAGES = 400


def loading_data(low_image_path, high_image_path):
    low_image = tf.io.read_file(low_image_path)
    low_image = tf.image.decode_png(low_image, channels=3)
    low_image = tf.image.resize(low_image, [IMAGE_SIZE, IMAGE_SIZE])
    low_image = low_image / 255.0

    high_image = tf.io.read_file(high_image_path)
    high_image = tf.image.decode_png(high_image, channels=3)
    high_image = tf.image.resize(high_image, [IMAGE_SIZE, IMAGE_SIZE])
    high_image = high_image / 255.0

    return low_image, high_image


def get_data(low_light_images, high_light_images):
    dataset = tf.data.Dataset.from_tensor_slices((low_light_images, high_light_images))
    dataset = dataset.map(lambda x, y: loading_data(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

train_low_light_images = sorted(glob("./train/low/*"))[:MAX_TRAIN_IMAGES]
train_high_light_images = sorted(glob("./train/high/*"))[:MAX_TRAIN_IMAGES]
val_low_light_images = sorted(glob("./train/low/*"))[MAX_TRAIN_IMAGES:]
val_high_light_images = sorted(glob("./train/high/*"))[MAX_TRAIN_IMAGES:]
test_low_light_images = sorted(glob("./test/low/*"))

#generating data from the train folder as train_dataset and val_dataset.
train_dataset = get_data(train_low_light_images, train_high_light_images)
val_dataset = get_data(val_low_light_images, val_high_light_images)

#making the EnlightenGAN model.
class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding="same", activation="relu"):
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, strides, padding)
        self.bn = layers.BatchNormalization()
        self.activation = layers.Activation(activation)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.activation(x)

class EnlightenGAN(keras.Model):
    def __init__(self):
        super(EnlightenGAN, self).__init__()
        self.encoder = keras.Sequential([
            ConvBlock(64, kernel_size=7, strides=2),
            ConvBlock(128, kernel_size=3, strides=2),
            ConvBlock(256, kernel_size=3, strides=2)
        ])
        self.decoder = keras.Sequential([
            layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(3, kernel_size=7, strides=2, padding="same"),
            layers.Activation("tanh")
        ])

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)

enlighten_gan = EnlightenGAN()
enlighten_gan.compile(optimizer='adam', loss='mean_squared_error')

history = enlighten_gan.fit(train_dataset, validation_data=val_dataset, epochs=100)

def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title(f"Train and Validation {item} Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()
#plotting the losses over epochs.
plot_result("loss")
#saving our model.
enlighten_gan.save("enlightengan_model.h5")

def infer(original_image):
    image = keras.preprocessing.image.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = enlighten_gan(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    return output_image
#getting images from test folder to get our predictions.
test_low_light_images = sorted(glob("./test/low/*"))
os.makedirs('./test/predicted/', exist_ok=True)

for img_path in test_low_light_images:
    original_image = Image.open(img_path)
    enhanced_image = infer(original_image)
    enhanced_image.save(os.path.join('./test/predicted/', os.path.basename(img_path)))




from skimage.metrics import peak_signal_noise_ratio as psnr
#calculating psnr
def calculate_psnr(original, enhanced):
    original = np.array(original, dtype=np.float32)
    enhanced = np.array(enhanced, dtype=np.float32)
    return psnr(original, enhanced, data_range=255)

psnr_scores = []

for img_path in train_low_light_images:
    original_image = Image.open(img_path)
    enhanced_image = infer(original_image)
    high_image = Image.open(img_path.replace("low", "high"))

    score = calculate_psnr(high_image, enhanced_image)
    psnr_scores.append(score)

average_psnr = np.mean(psnr_scores)
print(f'Average PSNR: {average_psnr}')
