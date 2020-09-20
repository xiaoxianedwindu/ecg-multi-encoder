"""
This script checks the installed tensorflow version and if the gpu is installed and seen by tensorflow.
"""

import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
