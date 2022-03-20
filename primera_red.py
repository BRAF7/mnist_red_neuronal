import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt

def resolucion_1():
    #labels = y determinada
    #Obtenemos datos para entrenar y para probar
    #Las imagenes son de 4 dimensiones
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    print(train_data.shape)
    # plt.imshow(train_data[0])

    model = tf.keras.models.Sequential()
    
    #*Se agregan capas
    #Las dimensiones pueden reducirse pero no aumentarse ya que no se tiene esa informacion
    #Agregamos 512 neuronas
    #usamos la funcion de activacion relu
    # resolucion de imagenes = 28x28
    model.add(tf.keras.layers.Dense(512,activation='relu', input_shape=(28*28,)))
    
    #Son 10 neuronas porque son 0-9 (10 posibles salidas)
    #Agregamos 10 neuronas
    #funcion de activacion softmax

    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    #Hacer dos funciones para trabajar con dos resoluciones
    #optimizer
    #loss = la perdida que existe
    #metrics
    #optimizador = https://interactivechaos.com/es/manual/tutorial-de-machine-learning/rmsprop
    #https://datasmarts.net/es/que-es-un-optimizador-y-para-que-se-usa-en-deep-learning/
    #loss = https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy
    #accuracy = https://keras.io/api/metrics/accuracy_metrics/
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    #Sirve para observar la informacion del modelo
    model.summary()


    #lo convertimos en un arreglo de dos dimensiones
    # de tamaño 28*28
    x_train = train_data.reshape((60000,28*28))
    
    #Se divide entre 255 porque es el valor maximo que puede tener cada pixel
    # de esta forma obtenemos valores entre 0-1
    x_train = x_train.astype('float32')/255

    x_test = test_data.reshape((10000,28*28))
    x_test = x_test.astype('float32')/255
    
    
    #Convertimos a un arreglo de 0 y 1 de tamaño 10
    # donde el 1 te marca que numero deberia ir
    #de esta forma se maneja mejor la informacion
    #de manera vectorial
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)

    #Guardamos el historial de los datos
    #comenzamos a entrenar
    #definimos 5 epocas
    #?Que es batch_size?
    # batch_size = https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
    historial = model.fit(x_train, y_train, epochs=5, batch_size=128)

    #Imprimimos segun la epoca
    print(model.fit(x_train, y_train, epochs=5, batch_size=128) )  

    #Graficamos la perdida que hubo

    return historial
    # print(model.evaluate(x_test, y_test))

def resolucion_2():
    #labels = y determinada
    #Obtenemos datos para entrenar y para probar
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    print(train_data.shape)
    # plt.imshow(train_data[0])

    model = tf.keras.models.Sequential()
    #Las dimensiones pueden reducirse pero no aumentarse ya que no se tiene esa informacion
    #Agregamos 512 neuronas
    #usamos la funcion de activacion relu
    # resolucion de imagenes = 28x28
    # relu no es una funcion lineal
    # nos devuelve un valor siempre que sea mayor a 0
    #caso contrario, devuelve 0
    model.add(tf.keras.layers.Dense(1000,activation='relu', input_shape=(28*28,)))
    #Agregamos 10 neuronas
    #funcion de activacion softmax
    #nos da la probabilidad de cada una de las posibles salidas
    model.add(tf.keras.layers.Dense(10,activation='softmax'))
    #Hacer dos funciones para trabajar con dos resoluciones
    #optimizer
    #loss = la funcion de perdida
    #categorical_crossentropy mide la distancia entre la prediccion real
    #y la prediccion del algoritmo
    #metrics
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()

    #Que estamos haciendo aqui?
    x_train = train_data.reshape((60000,28*28))
    x_train = x_train.astype('float32')/255

    x_test = test_data.reshape((10000,28*28))
    x_test = x_test.astype('float32')/255
    #Que estamos haciendo aqui?
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)

    #Guardamos el historial de los datos
    #comenzamos a entrenar
    #definimos 5 epocas
    #?Que es batch_size?
    historial = model.fit(x_train, y_train, epochs=5, batch_size=128)

    #Imprimimos segun la epoca
    print(model.fit(x_train, y_train, epochs=5, batch_size=128) )  

    #Graficamos la perdida que hubo
    return historial
    #sirve para medir la presicion de la red neuronal
    # print(model.evaluate(x_test, y_test))


def convolucional():
    # Model / data parameters
    #10 por las 10 neuronas
    num_classes = 10
    # tamaño de las imagenes
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    model = tf.keras.Sequential(
    [
        
        tf.keras.Input(shape=input_shape),
        #primera capa
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        #segunda capa
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # Hacemos un flatten para poder usar una red fully connected
        #https://anderfernandez.com/blog/que-es-una-red-neuronal-convolucional-y-como-crearlaen-keras/
        layers.Flatten(),
        #prevents overfitting
        #https://keras.io/api/layers/regularization_layers/dropout/
        layers.Dropout(0.5),
        # Añadimos una capa softmax para que podamos clasificar las imágenes
        layers.Dense(num_classes, activation="softmax"),
    ]
    )

    model.summary()
    batch_size = 128
    epochs = 5

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    historial = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return historial


if __name__ == '__main__':
    historial0 = convolucional()

    
    # historial1 = resolucion_1()
    # historial2 = resolucion_2()
    
    # plt.plot(historial1.history['loss'], label='Densa 1')
    # plt.legend()
    # plt.plot(historial2.history['loss'], label='Densa 2')
    # plt.legend()
    plt.plot(historial0.history['loss'], label='Convolucional')
    plt.legend()
    plt.xlabel("Epocas")
    plt.ylabel("Errores")
    plt.show()