import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
# init = tf.compat.v1.global_variables_initializer()
from ai_benchmark import AIBenchmark
print(tf.test.is_gpu_available())
results = AIBenchmark().run()
input()
