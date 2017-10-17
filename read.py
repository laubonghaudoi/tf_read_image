import glob
import os
import sys
from multiprocessing import Process, Queue, cpu_count

import cv2

FILENAMES_QUEUE = Queue(maxsize=10240)
IMAGE_QUEUE = Queue(maxsize=1024)
NUM_THREADS = cpu_count()

BASE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_PATH, "./data/")
IMAGE_FILES = sorted(glob.glob(os.path.join(DATA_PATH, "*.jpg")))


def decode():
    print("FILENAMEd_QUEUE get")
    filename = FILENAMES_QUEUE.get(block=True)

    image = cv2.imread(filename)
    print("IMAGE_QUEUE put")
    IMAGE_QUEUE.put({"filename": filename, "image": image})


def dequeue():
    print("IMAGE_QUEUE get")
    image_dict = IMAGE_QUEUE.get(block=True)
    print("image_dict")
    filename = image_dict["filename"]
    image = image_dict["image"]
    print(filename, image.size)


if __name__ == '__main__':
    for _ in range(NUM_THREADS):
        Process(target=decode, args=()).start()

    Process(target=dequeue, args=()).start()
    '''
    for filename in IMAGE_FILES:
        print("FILENAMES_QUEUE put")
        FILENAMES_QUEUE.put(filename)
    '''