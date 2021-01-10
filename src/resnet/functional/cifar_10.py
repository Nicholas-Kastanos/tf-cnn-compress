import tensorflow as tf 
from resnet.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt 
import numpy as np 
import os 

## parameters
batch_size = 32
epochs = 400
img_col, img_row = 32, 32
img_channels = 3

# Load cifar10 dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

Y_train = tf.keras.utils.to_categorical(Y_train, 10)
Y_test = tf.keras.utils.to_categorical(Y_test, 10)

# prepare data
mean_image = np.mean(X_train, axis=0)
X_train = X_train - mean_image
X_test = X_test - mean_image 
X_train = X_train/128. 
X_test = X_test/128. 

#####################################################

if not os.path.exists('data'):
    os.makedirs('data')

train_accuracies = np.load('data/train_accuracies.npy')
test_accuracies = np.load('data/test_accuracies.npy')
train_losses = np.load('data/train_losses.npy')
test_losses = np.load('data/test_losses.npy')

#####################################################

for filters in range(17, 71, 1):

    train_accuracies_ave = np.empty((0,1))
    test_accuracies_ave = np.empty((0,1))
    train_losses_ave = np.empty((0,1))
    test_losses_ave = np.empty((0,1))

    for run in range(0,3):
        model_dir = 'models/filters_' + str(filters) + '/run_' + str(run)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model = resnet_18(filters)
        model.build(input_shape=(None, img_row, img_col, img_channels))

        with open(model_dir + '/model_summary.txt', 'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        optimizer = tf.keras.optimizers.Adam()

        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=optimizer,
            metrics=['accuracy']
        )

        tensorboard_cb = tf.keras.callbacks.TensorBoard(
            log_dir='tensorboard/filters_' + str(filters) + '/run_' + str(run),
            histogram_freq=50
        )

        cbs = [tensorboard_cb]

        model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, Y_test),
            callbacks=cbs
        )

        model.save_weights(model_dir+'/parameters.h5')

        train_loss, train_acc = model.evaluate(X_train, Y_train)
        test_loss, test_acc = model.evaluate(X_test, Y_test)

        train_accuracies_ave = np.append(train_accuracies_ave, train_acc)
        train_losses_ave = np.append(train_losses_ave, train_loss)
        test_accuracies_ave = np.append(test_accuracies_ave, test_acc)
        test_losses_ave = np.append(test_losses_ave, test_loss)

    train_accuracies = np.append(train_accuracies, np.mean(train_accuracies_ave))
    train_losses = np.append(train_losses, np.mean(train_losses_ave))
    test_accuracies = np.append(test_accuracies, np.mean(test_accuracies_ave))
    test_losses = np.append(test_losses, np.mean(test_losses_ave))

    np.save('data/test_losses.npy', test_losses)
    np.save('data/train_losses.npy', train_losses)

    np.save('data/test_accuracies.npy', test_accuracies)
    np.save('data/train_accuracies.npy', train_accuracies)

    plt.plot(train_accuracies)
    plt.plot(test_accuracies)
    plt.legend(['Train', 'Test'])
    plt.savefig('data/accuracy.png')
    plt.close()

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(['Train', 'Test'])
    plt.savefig('data/loss.png')
    plt.close()
