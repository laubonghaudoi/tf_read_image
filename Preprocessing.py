import os, sys
import glob
from multiprocessing import cpu_count, Queue, Process
from PIL import Image

FILENAMES_QUEUE = Queue(maxsize=10240)
IMAGE_QUEUE = Queue(maxsize=1024)
NUM_THREADS = cpu_count()


def decode_image():
    while True:
        print ("FILENAMES_QUEUE GET")
        filename = FILENAMES_QUEUE.get(block=True)
        if os.path.exists(filename):
            image = Image.open(filename)
            print ("IMAGE_QUEUE PUT")
            IMAGE_QUEUE.put({"filename":filename, "image":image})
        else:
            print("file: %s not exists."% filename)

def dequeue_image():
    while True:
        print ("IMAGE_QUEUE GET")
        image_info = IMAGE_QUEUE.get(block=True)
        print ("IMAGE INFO")
        image = image_info["image"]
        filename = image_info["filename"]
        print (filename, image.size)


if __name__ == "__main__":
    if len(sys.argv)<2:
        print("image path required!")
        exit(1)
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print ("invalid image path.")
        exit(1)
    image_list = glob.glob(os.path.join(image_path, "*.jpg"))
    image_list += glob.glob(os.path.join(image_path, "*.png"))

    for _ in range(NUM_THREADS):
        Process(target=decode_image, args=()).start()

    Process(target=dequeue_image, args=()).start()

    for filename in image_list:
        print ("FILENAMES_QUEUE PUT")
        FILENAMES_QUEUE.put(filename)


