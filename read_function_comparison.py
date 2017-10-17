'''
Results show that tf.WholeFileReader().read(filename_queue)
is slightly faster than tf.read_file()
'''

import glob
import os
import time

import tensorflow as tf
from PIL import Image

BASE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_PATH, "./data/")
IMAGE_FILES = sorted(glob.glob(os.path.join(DATA_PATH, "*.jpg")))


def readFile():
    '''
    Dequeue the filename_queue and use tf.read_file to get image tensors
    '''
    with tf.name_scope("file_queue"):
        filename_queue = tf.train.string_input_producer(IMAGE_FILES)

        # This is where things are different
        filename = filename_queue.dequeue()
        img_read_op = tf.read_file(filename)

        # Decode jpeg files into numpy arrays
        img_decode_op = tf.image.decode_jpeg(img_read_op)

    # Start
    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # Create a coordinator
        coord = tf.train.Coordinator()
        # Launch the queue runner threads
        threads = tf.train.start_queue_runners(coord=coord)

        for _ in range(len(IMAGE_FILES)):
            img_array = sess.run(img_decode_op)  # img_decode_op.eval()
            #print(img_array.size)

        coord.request_stop()
        coord.join(threads)

    # Output run time
    end_time = time.time()
    return end_time - start_time


def wholeFileReader():
    '''
    Use tf.WholeFileReader().read(filename_queue) to get image tensors directly
    '''
    with tf.name_scope("file_queue"):
        filename_queue = tf.train.string_input_producer(IMAGE_FILES)

        # This is where things are different
        _, value = tf.WholeFileReader().read(filename_queue)

        # Decode jpeg files into numpy arrays
        img_decode_op = tf.image.decode_jpeg(value)

    # Start
    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # Create a coordinator
        coord = tf.train.Coordinator()
        # Launch the queue runner threads
        threads = tf.train.start_queue_runners(coord=coord)

        for _ in range(len(IMAGE_FILES)):
            img_array = sess.run(img_decode_op)  # img_decode_op.eval()
            #print(img_array.size)

        coord.request_stop()
        coord.join(threads)

    # Output run time
    end_time = time.time()
    return end_time - start_time

def imageOpen():
    start_time = time.time()
    for image_path in IMAGE_FILES:
        image = Image.open(image_path)
        print(image.getdata())
        
    end_time = time.time()
    return end_time - start_time


print(imageOpen())
print(readFile())
print(wholeFileReader())