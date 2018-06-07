# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mp

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """

    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h


num_classes = 3 #Num de clases
batch_size = 5  #Tamaño del lote de muestras

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):  #LISTA DE CAMINOS, TAMAÑO DEL LOTE

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):   #DEVUELVE UNA TUPLA (INDICE, PATH) -> i = indice, p = path // EL 'ENUMERATE' CREA LOS INDICES
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        #image, label = tf.image.decode_jpeg(file_image),  [float(i)]  # [one_hot(float(i), num_classes)] -> [float(i)] SON LOS LABELS

        image, label = tf.image.decode_jpeg(file_image, channels=1), one_hot(i, num_classes)
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140) #PARA PONER TODAS LAS IMAGENES AL MISMO TAMAÑO (imagen en formato 80x140)
        image = tf.reshape(image, [80, 140, 1]) #TENSODR DE 3 DIMENSIONES
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False): #x = batch correspondiente
    with tf.variable_scope('ConvNet', reuse=reuse):
        #MODELO QUE YA TENIAMOS (pero fuera de una funcion) -> VAMOS CONSTRUYENDO EL MODELO
        #ESTA FUNCION SOLO SE LLAMA 2 VECES, Y SOLO PARA CONSTRUIRLO, NO PARA EJECUTARLO
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * num_classes, 18 * 33 * 64]), units=5, activation=tf.nn.relu) #batch_size * num de neuronas (que será el num de clases)
        y = tf.layers.dense(inputs=h, units= num_classes, activation=tf.nn.softmax) #FUNCION DE ACTIVACION DE LA CAPA DE SALIDA (sigmoid por softmax y units= num_classes)
    return y #SALE LA CLASIFICACION QUE HA HECHO LA RED

#EL DATASOURCE DEVUELVE UN BATCH Y UNA ETIQUETA

example_batch_train, label_batch_train = dataSource(["data3/train/0/*.jpg", "data3/train/1/*.jpg", "data3/train/2/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["data3/valid/0/*.jpg", "data3/valid/1/*.jpg", "data3/valid/2/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["data3/test/0/*.jpg", "data3/test/1/*.jpg", "data3/test/2/*.jpg"], batch_size=batch_size)

"""
example_batch_train, label_batch_train = dataSource(["data3/train/0/*.jpg", "data3/train/1/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["data3/valid/0/*.jpg", "data3/valid/1/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["data3/test/0/*.jpg", "data3/test/1/*.jpg"], batch_size=batch_size)
"""

#SE COGE EL BATCH Y LA VALIDACION DE LAS LINEAS ANTERIORES Y SE METEN EN EL MODELO PARA QUE NOS DEVUELVA EL TENSOR
#REUSE (a true o false) ES LA REUTILIZACION -> a false para que cree una instancia nueva, y a true para que reutilice el anterior creado
example_batch_train_predicted = myModel(example_batch_train, reuse=False) #Tensor generado predicho con el batch anterior obtenido
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True) #Tensor generado predicho con la validacion anterior obtenido
example_batch_test_predicted = myModel(example_batch_test, reuse=True) #Tensor generado predicho con el test anterior obtenido

#TENSOR COST QUE SE MINIMIZARA EN LA LINEA 95
cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, tf.float32)))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) #OPTIMIZACION DEL COSTE -> SERA LLAMADA EN MUCHAS OCASIONES

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

errors = [] #Donde guardaremos los valores de los errores
error = 0   #Guardara el error calculado
epochs = [] #Donde guardaremos todas las epocas (sera el indice)
epoch = 0   #Guardara la epoca nueva
list_current_errors = [] #Lista de los errores actuales que se calculan
current_error = 0   #Guardara el valor del error actual calculado para luego almacenarlo en el array
umbral = 0.001 #Umbral para la condicion de parada
tolerancia = 10  #Tolerancia total
tolerancia_count = 0 #Contador comparador de la tolerancia para la parada de la red

with tf.Session() as sess:

    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for _ in range(400):    # _ son las epocas
        sess.run(optimizer)

        error = sess.run(cost)  #A partir del coste calculamos el error del entrenamiento
        errors.append(error)    #Lo añadimos a la lista
        current_error = sess.run(cost_valid)    #Calculamos el error actual a partir del coste acumulado (de validacion)
        list_current_errors.append(current_error)   #Lo añadimos
        epoch = _   #Se almacena la epoca en la que nos encontramos
        epochs.append(epoch)    #Guardamos dicha epoca en la lista

        if _ % 20 == 0: #LOS COSTES OPTIMIZADOS SE IMPRIMIRAN CADA 20 ITERACIONES
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(label_batch_valid))
            print(sess.run(example_batch_valid_predicted))
            print("Error en el entrenamiento: ", error) #AQUI SE VE EL ERROR DURANTE EL ENTRENAMIENTO
            print("Error actual: ", current_error)  #AQUI SE VE EL COSTE DE LA VALIDACION CALCULADO(ERROR ACTUAL)

        # EVALUAMOS LA CONDICIÓN DE PARADA DE LA RED
        # Calculamos el valor absoluto de la diferencia entre el error actual con el anterior y la comparamos con el umbral
        percentage = abs(list_current_errors[epoch] - list_current_errors[epoch - 1])
        if _ > 0 and percentage < umbral: #Si la epoca no es 0 y..
            tolerancia_count += 1  #.. la diferencia no supera el umbral, la tolerancia se incrementa
        else:  # Si la supera, no aumenta
            tolerancia_count = 0

        # Si se supera la tolerancia minima, es que la red no puede aprender mas (se ha estabilizado), por lo que
        # se finaliza el entrenamiento
        if tolerancia_count > tolerancia:
            print("FIN DEL ENTRENAMIENTO")
            break

    print("----------------------------")
    print("   GRAFICA DE EVOLUCION   ")
    print("----------------------------")
    mp.title("Evolucion Errores")  # Establecemos el titulo de la grafica
    mp.plot(epochs, list_current_errors, label='Error Actual')  # Etiqueta para cada error actual
    # mp.plot(epochs, errors, label = 'Entrenamiento de la red')  #Etiqueta para cada error durante el entrenamiento
    mp.xlabel('Epocas')  # Indicamos que el eje X serán las epocas
    mp.ylabel('Error')  # Indicamos que el eje Y serán los errores
    mp.legend()
    mp.show()

    print("----------------------")
    print("   Starting Test...   ")
    print("----------------------")
    test = sess.run(example_batch_test_predicted) #Se testea el lote de tests predicha
    y_test = sess.run(label_batch_test) #Se testea el lote de etiquetas (tests)

    precision = 0   #Calculamos la precision del test comparando los resultados anteriores
    for result_test, real in zip (test, y_test):
        if np.argmax(result_test) == np.argmax(real):
            precision += 1

    final_test = (precision / float(len(y_test))) * 100 #Sacamos su porcentaje y lo mostramos
    print ("La precision es de: ", final_test, "%")

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
