import matplotlib
matplotlib.use("Agg")

import os
import tensorflow as tf
from keras.models import Model
from keras.applications import MobileNetV2
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import BatchNormalization, MaxPooling2D, DepthwiseConv2D, Input
from keras.layers.pooling import AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import argparse
import pickle
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required= True, help= "path to input dataset")
ap.add_argument("-m", "--model", required= True, help= "path to the required model")
ap.add_argument("-l", "--label-bin", required= True, help= "path to the output label binarizer")
ap.add_argument("-e", "--epochs", required= True, type= int, default= 25, help= "# of epochs")
ap.add_argument("-p", "--plot", type=str, default= 'plot.png', help= "path to the accuracy/loss plots")
args = vars(ap.parse_args())


LABELS = ['badminton', 'cricket', 'football']

impath = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imgpath in impath:
    label = imgpath.split(os.path.sep)[-2]
    labels.append(label)
    image = cv2.imread(imgpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    data.append(image)

data = np.array(data)
labels = np.array(labels)

LB = LabelBinarizer()
labels = LB.fit_transform(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

trainaug = ImageDataGenerator(rotation_range=30,
                              zoom_range=0.15,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.15,
                              horizontal_flip=True,
                              fill_mode="nearest")

validaug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainaug.mean = mean
validaug.mean = mean

prtmodel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

i = prtmodel.output
#x = AveragePooling2D(pool_size=(7, 7))(i)
x = Flatten()(i)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(3, activation='softmax')(x)

model = Model(inputs=prtmodel.input, outputs=x)

for layer in prtmodel.layers:
    layer.trainable = False

opti = SGD(lr=1e-4, momentum=0.9, decay=1e-4/args["epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opti, metrics=["accuracy"])

H = model.fit_generator(trainaug.flow(trainX, trainY, batch_size=32), steps_per_epoch=len(trainX) // 32,
                              validation_data=validaug.flow(testX, testY), validation_steps=len(testX) // 32,
                              epochs=args["epochs"])

predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=LB.classes_))

N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="Train-Loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="Val-Loss")
plt.plot(np.arange(0, N), H.history["acc"], label="Accuracy")
plt.plot(np.arange(0, N), H.history["val_acc"], label="Val-Accuracy")
plt.title("Train/Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Losses/Accuracies")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

model.save(args["model"])

filename = open(args["label-bin"], "wb")
filename.write(pickle.dumps(LB))
filename.close()









