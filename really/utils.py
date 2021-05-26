from scipy import signal
import tensorflow as tf
import os
import numpy as np


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def dict_to_dataset(data_dict, batch_size=None):

    datasets = []
    for k in dict.keys():
        datasets.append(tf.data.Dataset.from_tensor_slices(data_dict[k]))
    dataset = tf.data.Dataset.zip(tuple(datasets))
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    keys = data_dict.keys()

    return dataset, keys


def dict_to_dict_of_datasets(data_dict, batch_size=None):

    dataset_dict = {}
    for k in data_dict.keys():
        dataset_dict[k] = tf.data.Dataset.from_tensor_slices(data_dict[k])
        if batch_size is not None:
            dataset_dict[k] = dataset_dict[k].batch(batch_size, drop_remainder=True)

    return dataset_dict


def all_subdirs_of(b="."):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

def tf_unique_2d(x):
    x_shape=x.get_shape() #(3,2)
    x1=tf.tile(x,[1,x_shape[0]]) #[[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]
    x2=tf.tile(x,[x_shape[0],1]) #[[1,2],[1,2],[1,2],[3,4],[3,4],[3,4]..]

    x1_2=tf.reshape(x1,[x_shape[0]*x_shape[0],x_shape[1]])
    x2_2=tf.reshape(x2,[x_shape[0]*x_shape[0],x_shape[1]])
    cond=tf.reduce_all(tf.equal(x1_2,x2_2),axis=1)
    cond=tf.reshape(cond,[x_shape[0],x_shape[0]]) #reshaping cond to match x1_2 & x2_2
    cond_shape=cond.get_shape()
    cond_cast=tf.cast(cond,tf.int32) #convertin condition boolean to int
    cond_zeros=tf.zeros(cond_shape,tf.int32) #replicating condition tensor into all 0's

    #CREATING RANGE TENSOR
    r=tf.range(x_shape[0])
    r=tf.add(tf.tile(r,[x_shape[0]]),1)
    r=tf.reshape(r,[x_shape[0],x_shape[0]])

    #converting TRUE=1 FALSE=MAX(index)+1 (which is invalid by default) so when we take min it wont get selected & in end we will only take values <max(indx).
    f1 = tf.multiply(tf.ones(cond_shape,tf.int32),x_shape[0]+1)
    f2 =tf.ones(cond_shape,tf.int32)
    cond_cast2 = tf.where(tf.equal(cond_cast,cond_zeros),f1,f2) #if false make it max_index+1 else keep it 1

    #multiply range with new int boolean mask
    r_cond_mul=tf.multiply(r,cond_cast2)
    r_cond_mul2=tf.reduce_min(r_cond_mul,axis=1)
    r_cond_mul3,unique_idx=tf.unique(r_cond_mul2)
    r_cond_mul4=tf.subtract(r_cond_mul3,1)

    #get actual values from unique indexes
    op=tf.gather(x,r_cond_mul4)

    return op
