# -*- coding:utf-8 -*-
import argparse
from res3d.drn import *
from keras.optimizers import SGD
from keras.utils import np_utils
from res3d.Schedules import onetenth_10_15_20
import numpy as np
import random
import cv2
import os
import matplotlib
import tensorflow as tf
matplotlib.use('AGG')
import matplotlib.pyplot as plt


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


def process_batch(batch_id, root_path, clip_length=16, random_length=False, train=True):
    batch_size = len(batch_id)
    batch = np.zeros((batch_size, 8, 112, 112, 3), dtype=np.float32)
    labels = np.zeros(batch_size, dtype='int')
    for i in range(batch_size):
        path = batch_id[i].split(' ')[0]
        label = int(batch_id[i].split(' ')[-1].strip())
        symbol = int(batch_id[i].split(' ')[1]) - 1
        imgs = os.listdir(root_path + path)
        imgs.sort(key=str.lower)
        if train:
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            is_flip = random.randint(0, 1)
            if random_length:
                is_16_24 = random.randint(0, 1)
                if is_16_24 == 0:
                    clip_length = 16
                else:
                    clip_length = 24
            for j in range(clip_length):
                if clip_length == 16:
                    if j % 2 == 0:
                        img = imgs[symbol + j]
                        image = cv2.imread(root_path + path + '/' + img)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (171, 128))
                        if is_flip == 1:
                            image = cv2.flip(image, 1)
                        batch[i][int(j/2)][:][:][:] = image[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
                elif clip_length == 24:
                    if j % 3 == 0:
                        img = imgs[symbol + j]
                        image = cv2.imread(root_path + path + '/' + img)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (171, 128))
                        if is_flip == 1:
                            image = cv2.flip(image, 1)
                        batch[i][int(j/3)][:][:][:] = image[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
            labels[i] = label
        else:
            for j in range(clip_length):
                if clip_length == 16:
                    if j % 2 == 0:
                        img = imgs[symbol + j]
                        image = cv2.imread(root_path + path + '/' + img)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (171, 128))
                        batch[i][int(j/2)][:][:][:] = image[8:120, 30:142, :]
                elif clip_length == 24:
                    if j % 3 == 0:
                        img = imgs[symbol + j]
                        image = cv2.imread(root_path + path + '/' + img)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (171, 128))
                        batch[i][int(j/3)][:][:][:] = image[8:120, 30:142, :]
            labels[i] = label
    return batch, labels


def preprocess(inputs):
    inputs = inputs.astype(np.float32)
    inputs[..., 0] -= 99.9
    inputs[..., 1] -= 92.1
    inputs[..., 2] -= 82.6
    inputs[..., 0] /= 65.8
    inputs[..., 1] /= 62.3
    inputs[..., 2] /= 60.3
    inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
    return inputs


def generator_train_batch(train_txt, batch_size, num_classes, img_path):
    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(num // batch_size):
            a = i * batch_size
            b = (i + 1) * batch_size
            x_train, x_labels = process_batch(new_line[a:b], img_path, clip_length=16, random_length=False)
            x = preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            yield x, y


def generator_val_batch(val_txt, batch_size, num_classes, img_path):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(num // batch_size):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test, y_labels = process_batch(new_line[a:b], img_path, clip_length=16,random_length=False, train=False)
            x = preprocess(y_test)
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


def main():
    image_path = '/Users/chenbingxu/PycharmProjects/c3d/ucfimgs/'
    train_file = '/Users/chenbingxu/PycharmProjects/c3d/ucfTrainTestlist/train_list.txt'
    test_file = '/Users/chenbingxu/PycharmProjects/c3d/ucfTrainTestlist/test_list.txt'
    f1 = open(train_file, 'r')
    f2 = open(test_file, 'r')
    lines = f1.readlines()
    f1.close()
    train_samples = len(lines)
    lines = f2.readlines()
    f2.close()
    val_samples = len(lines)

    num_classes = 6
    batch_size = 32
    epochs = 2

    model = Residual_DenseNet(num_classes, (112, 112, 8, 3), dropout_rate=0.2)
    lr = 0.01
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
    model.summary()
    history = model.fit_generator(generator_train_batch(train_file, batch_size, num_classes, image_path),
                                  steps_per_epoch=train_samples // batch_size,
                                  epochs=epochs,
                                  callbacks=[onetenth_10_15_20(lr)],
                                  validation_data=generator_val_batch(test_file,
                                                                      batch_size, num_classes, image_path),
                                  validation_steps=val_samples // batch_size,
                                  verbose=1)
    if not os.path.exists('results/'):
        os.mkdir('results/')
    plot_history(history, 'results/')
    save_history(history, 'results/')
    model.save_weights('results/weights_drn3d.h5')
    writer = tf.summary.FileWriter("./log", tf.get_default_graph())
    writer.close()

if __name__ == '__main__':
    main()