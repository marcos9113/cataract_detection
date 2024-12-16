import pandas as pd
import numpy as np
import os
import scipy

image_paths = {
    "train_cataract" : [],
    "train_normal" : [],
    "test_cataract" : [],
    "test_normal" : []
}


for dirname, _, filenames in os.walk('/home/marcos_007/Documents/Atria/Cognitive AI/Final Project/CNN/dataset/'):
    for filename in filenames:
        path = os.path.join(dirname, filename)
        # print(path)
        if "train/cataract" in path:
            image_paths["train_cataract"].append(path)
        elif "train/normal" in path:
            image_paths["train_normal"].append(path)
        elif "test/cataract" in path:
            image_paths["test_cataract"].append(path)
        elif "test/normal" in path:
            image_paths["test_normal"].append(path)


from PIL import Image
from matplotlib import pyplot as plt


# sample_img = np.array(Image.open(image_paths["test_normal"][2]))
# print(f"size of image : {np.shape(sample_img)}")
# plt.imshow(sample_img)
#
# sample_img = np.array(Image.open(image_paths["test_cataract"][0]))
# print(f"size of image : {np.shape(sample_img)}")
# plt.imshow(sample_img)

training_dir = "/home/marcos_007/Documents/Atria/Cognitive AI/Final Project/CNN/dataset/train"
image_size = (55, 94, 3)
target_size = (55, 94)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size = target_size,
    class_mode = 'binary'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers.experimental import RMSprop

model = Sequential([
    Conv2D(16, (3,3), activation='relu',input_shape=image_size),
    MaxPooling2D(2, 2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])


# model.summary()

model.compile(
    loss = 'binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['accuracy']
)


history = model.fit_generator(
    train_generator,
    epochs=15
)


# epochs = range(1, 16)
# plt.figure(figsize=(10, 5))
# plt.title("loss vs accuracy of model")
# plt.plot(epochs, history.history['loss'], label='loss')
# plt.plot(epochs, history.history['accuracy'], label='accuracy')
# plt.grid()
# plt.xlabel("epochs")
# plt.grid()
# plt.legend()


import tensorflow as tf

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to .h5 format

model.save("model.h5")


from keras.preprocessing import image

label = train_generator.class_indices
# print(label)

path = image_paths["test_cataract"][30]
img = Image.open(path)
plt.imshow(np.array(img))
img = np.array(img.resize((94, 55)))
img = np.expand_dims(img, axis=0)
pred = model.predict(img)
print(f"predicted class : {'normal' if pred[0] > 0.5 else 'cataract'}")
