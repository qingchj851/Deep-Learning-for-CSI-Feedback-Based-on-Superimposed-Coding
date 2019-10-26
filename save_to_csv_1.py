import tensorflow as tf
import numpy as np
import pandas as pd
from my_function import batch_norm
from my_function import despreading
from my_function import spreading

sess = tf.Session()

file_name = "./t/Unit2-model-12000"
CSI_len = 64   # Length of the DL-CSI
L_len = 512   # Length of the UL-US
Ek = 10**(0.5)     # Sending power
rou = 0.2     # Power proportional coefficient
batch_size = 200
std = 0.01    # Weight initializes variance
lr = 0.0001    # Learning rate
theta = 0.00002 # Regularization coefficient

# Network structure parameter:
H_Net_1 = np.array([2*CSI_len, 16*CSI_len, 2*CSI_len])
D_Net_1 = np.array([2*L_len,16*L_len,2*L_len])

Const_h = np.float32((rou/CSI_len)**0.5)
Const_d = np.float32(((1-rou))**0.5)
Const_dd = np.float32(Const_d**(-1))
walsh = pd.read_csv('walsh_64_512.csv')  # 1024x32
walsh = walsh.astype(np.float32)




def H_DNN_1(input):
    w_1 = tf.Variable(tf.random_normal([H_Net_1[0],H_Net_1[1]],mean=0.0,stddev=std))
    b_1 = tf.Variable(tf.zeros([1,H_Net_1[1]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
    layer_1 = tf.nn.swish((tf.add(tf.matmul((input), w_1), b_1)))
    w_2 = tf.Variable(tf.random_normal([H_Net_1[1],H_Net_1[2]],mean=0.0,stddev=std))
    b_2 = tf.Variable(tf.zeros([1,H_Net_1[2]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
    layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
    return [layer_2,w_1,b_1,w_2,b_2]
# def H_DNN_1(input):
#     w_1 = np.float32(pd.read_csv('wh_11.csv').ix[:,1:])
#     b_1 = np.float32(pd.read_csv('bh_11.csv').ix[:,1:])
#     layer_1 = tf.nn.swish((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = np.float32(pd.read_csv('wh_12.csv').ix[:,1:])
#     b_2 = np.float32(pd.read_csv('bh_12.csv').ix[:,1:])
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return [layer_2,w_1,b_1,w_2,b_2]


# def D_DNN_1(input):
    # w_1 = tf.Variable(tf.random_normal([D_Net_1[0],D_Net_1[1]],mean=0.0,stddev=std))
    # b_1 = tf.Variable(tf.zeros([1,D_Net_1[1]]))
    # tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
    # tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
    # layer_1 = tf.nn.swish((tf.add(tf.matmul((input), w_1), b_1)))
    # w_2 = tf.Variable(tf.random_normal([D_Net_1[1],D_Net_1[2]],mean=0.0,stddev=std))
    # b_2 = tf.Variable(tf.zeros([1,D_Net_1[2]]))
    # tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
    # tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
    # layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
    # return [layer_2,w_1,b_1,w_2,b_2]
# def D_DNN_1(input):
#     w_1 = np.float32(pd.read_csv('wd_11.csv').ix[:,1:])
#     b_1 = np.float32(pd.read_csv('bd_11.csv').ix[:,1:])
#     layer_1 = tf.nn.swish((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = np.float32(pd.read_csv('wd_12.csv').ix[:,1:])
#     b_2 = np.float32(pd.read_csv('bd_12.csv').ix[:,1:])
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return [layer_2,w_1,b_1,w_2,b_2]


# def H_DNN_2(input):
#     w_1 = tf.Variable(tf.random_normal([H_Net_1[0],H_Net_1[1]],mean=0.0,stddev=std))
#     b_1 = tf.Variable(tf.zeros([1,H_Net_1[1]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
#     layer_1 = tf.nn.swish((tf.add(tf.matmul((input), w_1), b_1)))
#     w_2 = tf.Variable(tf.random_normal([H_Net_1[1],H_Net_1[2]],mean=0.0,stddev=std))
#     b_2 = tf.Variable(tf.zeros([1,H_Net_1[2]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return [layer_2,w_1,b_1,w_2,b_2]
# def H_DNN_2(input):
#     w_1 = np.float32(pd.read_csv('wh_21.csv').ix[:,1:])
#     b_1 = np.float32(pd.read_csv('bh_21.csv').ix[:,1:])
#     layer_1 = tf.nn.swish((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = np.float32(pd.read_csv('wh_22.csv').ix[:,1:])
#     b_2 = np.float32(pd.read_csv('bh_22.csv').ix[:,1:])
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return [layer_2,w_1,b_1,w_2,b_2]


# def D_DNN_2(input):
#     w_1 = tf.Variable(tf.random_normal([D_Net_1[0],D_Net_1[1]],mean=0.0,stddev=std))
#     b_1 = tf.Variable(tf.zeros([1,D_Net_1[1]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
#     layer_1 = tf.nn.swish((tf.add(tf.matmul((input), w_1), b_1)))
#     w_2 = tf.Variable(tf.random_normal([D_Net_1[1],D_Net_1[2]],mean=0.0,stddev=std))
#     b_2 = tf.Variable(tf.zeros([1,D_Net_1[2]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return [layer_2,w_1,b_1,w_2,b_2]
# def D_DNN_2(input):
#     w_1 = np.float32(pd.read_csv('wd_21.csv').ix[:,1:])
#     b_1 = np.float32(pd.read_csv('bd_21.csv').ix[:,1:])
#     layer_1 = tf.nn.swish((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = np.float32(pd.read_csv('wd_22.csv').ix[:,1:])
#     b_2 = np.float32(pd.read_csv('bd_22.csv').ix[:,1:])
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return [layer_2,w_1,b_1,w_2,b_2]





xs = tf.placeholder(tf.float32, shape=[None, 2 * L_len])
target_H = tf.placeholder(tf.float32, shape=[None, 2 * CSI_len])
target_D = tf.placeholder(tf.float32, shape=[None, 2 * L_len])

h_1 = despreading(xs)
[H_1,wh_11,bh_11,wh_12,bh_12] = H_DNN_1(h_1)
# d_1 = tf.subtract(xs, tf.multiply(Const_h, spreading(H_1)))
# [D_1,wd_11,bd_11,wd_12,bd_12] = D_DNN_1(d_1)
# h_2 = despreading(tf.subtract(xs, tf.multiply(Const_d, D_1)))
# [H_2,wh_21,bh_21,wh_22,bh_22] = H_DNN_2(h_2)
# d_2 = tf.subtract(xs, tf.multiply(Const_h, spreading(H_2)))
# [D_2,wd_21,bd_21,wd_22,bd_22] = D_DNN_2(d_2)



saver = tf.train.Saver()
saver.restore(sess, file_name)

df_wh_11 = pd.DataFrame(sess.run(wh_11))
df_bh_11 = pd.DataFrame(sess.run(bh_11))
df_wh_12 = pd.DataFrame(sess.run(wh_12))
df_bh_12 = pd.DataFrame(sess.run(bh_12))

# df_wd_11 = pd.DataFrame(sess.run(wd_11))
# df_bd_11 = pd.DataFrame(sess.run(bd_11))
# df_wd_12 = pd.DataFrame(sess.run(wd_12))
# df_bd_12 = pd.DataFrame(sess.run(bd_12))

# df_wh_21 = pd.DataFrame(sess.run(wh_21))
# df_bh_21 = pd.DataFrame(sess.run(bh_21))
# df_wh_22 = pd.DataFrame(sess.run(wh_22))
# df_bh_22 = pd.DataFrame(sess.run(bh_22))

# df_wd_21 = pd.DataFrame(sess.run(wd_21))
# df_bd_21 = pd.DataFrame(sess.run(bd_21))
# df_wd_22 = pd.DataFrame(sess.run(wd_22))
# df_bd_22 = pd.DataFrame(sess.run(bd_22))

df_wh_11.to_csv('wh_11.csv')
df_bh_11.to_csv('bh_11.csv')
df_wh_12.to_csv('wh_12.csv')
df_bh_12.to_csv('bh_12.csv')

# df_wd_11.to_csv('wd_11.csv')
# df_bd_11.to_csv('bd_11.csv')
# df_wd_12.to_csv('wd_12.csv')
# df_bd_12.to_csv('bd_12.csv')

# df_wh_21.to_csv('wh_21.csv')
# df_bh_21.to_csv('bh_21.csv')
# df_wh_22.to_csv('wh_22.csv')
# df_bh_22.to_csv('bh_22.csv')

# df_wd_21.to_csv('wd_21.csv')
# df_bd_21.to_csv('bd_21.csv')
# df_wd_22.to_csv('wd_22.csv')
# df_bd_22.to_csv('bd_22.csv')

