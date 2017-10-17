'''
Time comparison of 3 methods to read image files

'''

import glob
import os
import time

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = ""

BASE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_PATH, "./data/")
LOG_PATH = os.path.join(BASE_PATH, "./log/")
IMAGE_FILES = sorted(glob.glob(os.path.join(DATA_PATH, "*.jpg")))


def read_file():
    '''
    ### `tf.read_file()` and `tf.image.decode_jpeg()`
    '''
    filename_queue = tf.train.string_input_producer(
        IMAGE_FILES, num_epochs=None, shuffle=False)
    filename = filename_queue.dequeue()

    img_read_op = tf.read_file(filename)
    img_decode_op = tf.image.decode_jpeg(img_read_op)

    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for file in IMAGE_FILES:
            print("{}:{}".format(file, img_decode_op.eval().size))
        print('\n')
        coord.request_stop()
        coord.join(threads)

    end_time = time.time()
    return end_time - start_time


def WholeFileReader():
    '''
    ### `tf.WholeFileReader().read(filename_queue)` and `tf.image.decode(value)`
    '''
    filename_queue = tf.train.string_input_producer(IMAGE_FILES, shuffle=False)

    key, value = tf.WholeFileReader().read(filename_queue)
    img_decode_op = tf.image.decode_jpeg(value)

    # Start
    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for file in IMAGE_FILES:
            print("{}:{}".format(file, img_decode_op.eval().size))
        print('\n')
        coord.request_stop()
        coord.join(threads)

    end_time = time.time()
    return end_time - start_time


def read_file_FIFOdecode(capacity=32, number_threads=1):
    '''
    ### `tf.read_file()` and `tf.FIFOQueue()`the`tf.image.decode_jpeg()`
    '''
    filename_queue = tf.train.string_input_producer(IMAGE_FILES, shuffle=False)
    filename = filename_queue.dequeue()

    img_read_op = tf.read_file(filename)
    img_decode_op = tf.image.decode_jpeg(img_read_op)

    img_decode_queue = tf.FIFOQueue(capacity=capacity, dtypes=tf.uint8)
    enqueue_decode_op = img_decode_queue.enqueue(img_decode_op)
    img = img_decode_queue.dequeue()

    # Create a queue runner to run the image decode queue
    queue_runner = tf.train.QueueRunner(img_decode_queue,
                                        [enqueue_decode_op] * number_threads,
                                        img_decode_queue.close(),
                                        img_decode_queue.close(cancel_pending_enqueues=True))
    tf.train.add_queue_runner(queue_runner)

    # Start
    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Start
        for file in IMAGE_FILES:
            print("{}:{}".format(file, img.eval().size))
        print('\n')
        coord.request_stop()
        coord.join(threads)

    # Output run time
    end_time = time.time()
    return end_time - start_time


def WholeFileReader_FIFOdecode(capacity=32, number_threads=4):
    '''
    ### `tf.WholeFileReader().read()` and `tf.FIFOQueue()`the`tf.image.decode_jpeg()`
    '''
    filename_queue = tf.train.string_input_producer(IMAGE_FILES, shuffle=False)

    key, value = tf.WholeFileReader().read(filename_queue)
    img_decode_op = tf.image.decode_jpeg(value)

    img_decode_queue = tf.FIFOQueue(capacity=capacity, dtypes=tf.uint8)
    enqueue_decode_op = img_decode_queue.enqueue(img_decode_op)
    img = img_decode_queue.dequeue()

    # Create a queue runner to run the image decode queue
    queue_runner = tf.train.QueueRunner(img_decode_queue,
                                        [enqueue_decode_op] * number_threads,
                                        img_decode_queue.close(),
                                        img_decode_queue.close(cancel_pending_enqueues=True))
    tf.train.add_queue_runner(queue_runner)

    # Start
    start_time = time.time()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Start
        for file in IMAGE_FILES:
            print("{}:{}".format(file, img.eval().size))
        print('\n')
        coord.request_stop()
        coord.join(threads)

    # Output run time
    end_time = time.time()
    return end_time - start_time


if __name__ == '__main__':
    results = []
    results.append(read_file())
    results.append(WholeFileReader())
    results.append(read_file_FIFOdecode())
    results.append(WholeFileReader_FIFOdecode())

    print("tf.read_file(): {} seconds.".format(results[0]))
    print("tf.WholeFileReader(): {} seconds.".format(results[1]))
    print("tf.read_file() FIFOQueue decode: {} seconds.".format(results[2]))
    print(
        "tf.WholeFileReader() FIFOQueue decode: {} seconds.".format(results[3]))
