import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

y_train = y_train.reshape(-1,)
y_test  = y_test.reshape(-1,)


cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def show_image_cifar10(x, y, index):
    plt.imshow(x[index])
    plt.xlabel(cifar10_classes[y[index]])
    
show_image_cifar10(x_train, y_train, 11)


x_train = x_train/255 
x_test = x_test/255


# ## CNN Model

cnn = models.Sequential([
    #cnn
    layers.Conv2D(filters = 16, kernel_size = (3, 3), activation = 'relu', input_shape = (32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'),
    layers.MaxPooling2D((2, 2)),
    
    #dense
    layers.Flatten(),
    layers.Dense(32, activation = 'relu'),
    layers.Dense(2, activation = 'softmax'),
])


cnn.compile(  optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = 'accuracy')


# ## Training the model
cnn.fit(x_train, y_train, epochs = 5)
cnn.evaluate(x_test, y_test)


# ## Testing the model
pred = cnn.predict(x_test)
pred_result = [np.argmax(element) for element in pred]

show_image_ds(x_test, pred_result, 1)
tf.keras.models.save_model(cnn,'cnn_model.hdf5')
