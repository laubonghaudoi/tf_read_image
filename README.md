# Run time comparison test of reading image files

To run the test:

1. Create a `./data` folder in this directory and put any test images (`.jpeg` files) in it. It is recommended to put more than 100 images to see a better comparison outcome.
1. Run the commmand:  `python ./queue_read_experiment/queue_read_comparison.py`

## Methods

- __Baseline__: Use a `for` loop to read files
- __Single queue__: Use `tf.train.string_input_producer` to enqueue file names with ONE thread and use `tf.WholeFileReader` to dequeue images
- __Multi queue__: Use `tf.train.string_input_producer` to enqueue file names and `tf.FIFOQueue` to enqueue jpeg decode operation