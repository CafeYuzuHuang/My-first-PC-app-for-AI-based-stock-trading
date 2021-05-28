# -*- coding: utf-8 -*-

import tensorflow as tf
import timeit


# mat(a, c) = mat(a, b) x mat(b, c)
a = 10000
b = 5000
c = 1000
n = 10
# n = 1

with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([a, b])
    cpu_b = tf.random.normal([b, c])
    print(" ***      *** ")
    print(cpu_a.device, cpu_b.device)

with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([a, b])
    gpu_b = tf.random.normal([b, c])
    print(" ***      *** ")
    print(gpu_a.device, gpu_b.device)

# Perform matrix mulplication
def cpu_run():
    with tf.device('/cpu:0'):
        res = tf.matmul(cpu_a, cpu_b)
    return res

def gpu_run():
    with tf.device('/gpu:0'):
        res = tf.matmul(gpu_a, gpu_b)
    return res

print(" ***      *** ")
cpu_time = timeit.timeit(cpu_run, number = n)
gpu_time = timeit.timeit(gpu_run, number = n)
print('Warmup time',cpu_time,gpu_time)

cpu_time = timeit.timeit(cpu_run, number = n)
gpu_time = timeit.timeit(gpu_run, number = n)
print('Run time',cpu_time,gpu_time)

# Done!!
