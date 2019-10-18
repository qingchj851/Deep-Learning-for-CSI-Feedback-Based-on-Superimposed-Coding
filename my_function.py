import tensorflow as tf
import numpy as np
import pandas as pd
import copy

file_name = "./t/Unit2-model-16000"
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
walsh = pd.read_csv('walsh_64_512.csv')  # 512x64
walsh = walsh.astype(np.float32)
m = 10000
leng = 15
epoch = 2000
iteration = int(epoch*leng*m/batch_size + 1)


def spreading_np(x):
    Q = walsh
    out = np.dot(x,np.transpose(Q))
    return out


def Noise(m):
    noise_temp = []
    for ii in range(m):
        G = ((CSI_len)**(-0.5))*(0.5**0.5)*(np.random.normal(0,1,[CSI_len,1])+1j*np.random.normal(0,1,[CSI_len,1]))
        N_mat = (Ek ** -0.5) * (0.5**0.5)*(np.random.normal(0,1,[CSI_len,L_len])+1j*np.random.normal(0,1,[CSI_len,L_len]))
        N_temp = np.dot(np.linalg.pinv(G),N_mat)
        noise_temp.append(N_temp)
    return np.reshape(noise_temp,[m,-1])

def batch_norm(x):
    y = copy.copy(x)
    mean = tf.reduce_mean(y)
    y = (y - mean) / tf.sqrt(tf.reduce_mean(tf.square(y - mean)))
    return y



def despreading(x):
    y = tf.reshape(tf.stack([tf.matmul(x[:,:L_len],walsh),tf.matmul(x[:,L_len:],walsh)],axis=1),[-1,2*CSI_len])
    return y


def spreading(x):
    Real = tf.matmul(x[:,:CSI_len],np.transpose(walsh))
    Imag = tf.matmul(x[:,CSI_len:],np.transpose(walsh))
    return tf.reshape(tf.stack([Real,Imag],axis=1),[-1,2*L_len])


def data_map(x_):
    return np.where(x_>0,1.,0.)


def BER(x,y):
    num_x = np.size(x)
    temp = x-y
    num_temp = sum(sum(temp**2))
    return  num_temp/num_x


def sig_gen(M,N):
    data = (np.random.randint(0,2,[M,N])*2. - 1.) + 1j*(np.random.randint(0,2,[M,N])*2. - 1.)
    out = np.sqrt(1/2)*data
    return out



def H_DNN_1(input):
    w_1 = tf.Variable(tf.random_normal([H_Net_1[0],H_Net_1[1]],mean=0.0,stddev=std))
    b_1 = tf.Variable(tf.zeros([1,H_Net_1[1]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
    layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
    w_2 = tf.Variable(tf.random_normal([H_Net_1[1],H_Net_1[2]],mean=0.0,stddev=std))
    b_2 = tf.Variable(tf.zeros([1,H_Net_1[2]]))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
    tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
    layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
    return layer_2
# def H_DNN_1(input):
#     w_1 = np.float32(pd.read_csv('wh_11.csv').ix[:,1:])
#     b_1 = np.float32(pd.read_csv('bh_11.csv').ix[:,1:])
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = np.float32(pd.read_csv('wh_12.csv').ix[:,1:])
#     b_2 = np.float32(pd.read_csv('bh_12.csv').ix[:,1:])
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return layer_2


# def D_DNN_1(input):
#     w_1 = tf.Variable(tf.random_normal([D_Net_1[0],D_Net_1[1]],mean=0.0,stddev=std))
#     b_1 = tf.Variable(tf.zeros([1,D_Net_1[1]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = tf.Variable(tf.random_normal([D_Net_1[1],D_Net_1[2]],mean=0.0,stddev=std))
#     b_2 = tf.Variable(tf.zeros([1,D_Net_1[2]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return layer_2
# def D_DNN_1(input):
#     w_1 = np.float32(pd.read_csv('wd_11.csv').ix[:,1:])
#     b_1 = np.float32(pd.read_csv('bd_11.csv').ix[:,1:])
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = np.float32(pd.read_csv('wd_12.csv').ix[:,1:])
#     b_2 = np.float32(pd.read_csv('bd_12.csv').ix[:,1:])
#     layer_2 = 0.5**0.5*tf.nn.tanh(10000*(tf.add(tf.matmul((layer_1),w_2),b_2)))
#     return layer_2


# def H_DNN_2(input):
#     w_1 = tf.Variable(tf.random_normal([H_Net_1[0],H_Net_1[1]],mean=0.0,stddev=std))
#     b_1 = tf.Variable(tf.zeros([1,H_Net_1[1]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = tf.Variable(tf.random_normal([H_Net_1[1],H_Net_1[2]],mean=0.0,stddev=std))
#     b_2 = tf.Variable(tf.zeros([1,H_Net_1[2]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return layer_2
# def H_DNN_2(input):
#     w_1 = np.float32(pd.read_csv('wh_21.csv').ix[:,1:])
#     b_1 = np.float32(pd.read_csv('bh_21.csv').ix[:,1:])
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = np.float32(pd.read_csv('wh_22.csv').ix[:,1:])
#     b_2 = np.float32(pd.read_csv('bh_22.csv').ix[:,1:])
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return layer_2


# def D_DNN_2(input):
#     w_1 = tf.Variable(tf.random_normal([D_Net_1[0],D_Net_1[1]],mean=0.0,stddev=std))
#     b_1 = tf.Variable(tf.zeros([1,D_Net_1[1]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_1)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_1)
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = tf.Variable(tf.random_normal([D_Net_1[1],D_Net_1[2]],mean=0.0,stddev=std))
#     b_2 = tf.Variable(tf.zeros([1,D_Net_1[2]]))
#     tf.add_to_collection(tf.GraphKeys.WEIGHTS,w_2)
#     tf.add_to_collection(tf.GraphKeys.BIASES, b_2)
#     layer_2 = (tf.add(tf.matmul((layer_1),w_2),b_2))
#     return layer_2
# def D_DNN_2(input):
#     w_1 = np.float32(pd.read_csv('wd_21.csv').ix[:,1:])
#     b_1 = np.float32(pd.read_csv('bd_21.csv').ix[:,1:])
#     layer_1 = tf.nn.tanh((tf.add(tf.matmul(batch_norm(input), w_1), b_1)))
#     w_2 = np.float32(pd.read_csv('wd_22.csv').ix[:,1:])
#     b_2 = np.float32(pd.read_csv('bd_22.csv').ix[:,1:])
#     layer_2 = 0.5**0.5*tf.nn.tanh(10000*tf.add(tf.matmul((layer_1),w_2),b_2))
#     return layer_2




xs = tf.placeholder(tf.float32, shape=[None, 2 * L_len])
target_H = tf.placeholder(tf.float32, shape=[None, 2 * CSI_len])
target_D = tf.placeholder(tf.float32, shape=[None, 2 * L_len])

h_1 = despreading(xs)
H_1 = H_DNN_1(h_1)
# d_1 = tf.subtract(xs, tf.multiply(Const_h, spreading(H_1)))
# D_1 = D_DNN_1(d_1)
# h_2 = despreading(tf.subtract(xs, tf.multiply(Const_d, D_1)))
# H_2 = H_DNN_2(h_2)
# d_2 = tf.subtract(xs, tf.multiply(Const_h, spreading(H_2)))
# D_2 = D_DNN_2(d_2)


#*******************************Generate the training data set*****************************************
def gen_data():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    for ii in range(leng):
        L_1 = sig_gen(m, L_len)  # UL-US data
        CSI = 0.5 ** 0.5 * (np.random.normal(0, 1, [m, CSI_len]) + 1j * np.random.normal(0, 1, [m, CSI_len]))  # DL-CSI
        CSI_spreading = spreading_np(CSI)
        Trans_data = np.sqrt(1 - rou) * L_1 + np.sqrt(rou / CSI_len) * CSI_spreading  # Transmit data
        noise = Noise(m)
        Rece_data = Trans_data + noise  #  Receive data

        # Complex Numbers are converted to real Numbers:
        R_real_value = np.hstack((np.real(Rece_data), np.imag(Rece_data)))
        CSI_real_value = (np.hstack((np.real(CSI), np.imag(CSI))))
        L_real_value = (np.hstack((np.real(L_1), np.imag(L_1))))

        # Save the training data set:
        x_val = pd.DataFrame(R_real_value)
        y_CSI = pd.DataFrame(CSI_real_value)
        y_data = pd.DataFrame(L_real_value)

        df1 = df1.append(x_val)
        df2 = df2.append(y_CSI)
        df3 = df3.append(y_data)
        print(ii)
    print(np.shape(df1))
    print(np.shape(df2))
    print(np.shape(df3))
    print("Save data: Rece_data")
    df1.to_csv('Rece_data.csv')
    print("Save data: DL_CSI")
    df2.to_csv('DL_CSI.csv')
    print("Save data: UL_US")
    df3.to_csv('UL_US.csv')

#*************************Training*********************************************
def train():
    regularizer = tf.contrib.layers.l2_regularizer(theta)  #  Regularization
    reg_term = tf.contrib.layers.apply_regularization(regularizer)

    loss_H = tf.reduce_mean(tf.square(target_H - H_1))/tf.reduce_mean(tf.square(target_H))
    # loss_D = tf.reduce_mean(tf.square(target_D - D_1))/tf.reduce_mean(tf.square(target_D))
    loss = loss_H + reg_term
    # loss = loss_D + reg_term
    train = tf.train.AdamOptimizer(lr).minimize(loss)


    x_ = pd.read_csv('x_val.csv')
    H_ = pd.read_csv('y_CSI.csv')
    D_ = pd.read_csv('y_data.csv')



    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


    saver = tf.train.Saver(max_to_keep=4)
    for ii in range(iteration):
        rand_index = np.random.choice(len(x_), size=batch_size)
        x_input = x_.ix[rand_index, 1:]
        H_output = H_.ix[rand_index, 1:]
        D_output = D_.ix[rand_index, 1:]
        feed_dict = {xs: x_input, target_H: H_output, target_D: D_output}
        sess.run(train, feed_dict=feed_dict)

        if ii % 1000 == 0:
            # out_data = data_map(sess.run(D_1, feed_dict=feed_dict))
            # ber = BER(out_data, data_map(sess.run(target_D, feed_dict=feed_dict)))
            print('-' * 50)
            print('After %d iterations，loss_H = %8f.' % (ii, sess.run(loss_H, feed_dict=feed_dict)))
            # print('After %d iterations，loss_D = %8f.' % (ii, sess.run(loss_D, feed_dict=feed_dict)))
            # print('After %d iterations，BER = %12f.' % (ii,ber))
            saver.save(sess, 't/Unit2-model', global_step=ii)


#******************************Test**************************************************
def test_model():
    L_1 = sig_gen(m, L_len)  # UL-US data
    CSI = 0.5 ** 0.5 * (np.random.normal(0, 1, [m, CSI_len]) + 1j * np.random.normal(0, 1, [m, CSI_len]))  # DL-CSI
    CSI_spreading = spreading_np(CSI)
    Trans_data = np.sqrt(1 - rou) * L_1 + np.sqrt(rou / CSI_len) * CSI_spreading  # Transmit data
    noise = Noise(m)
    Rece_data = Trans_data + noise  # Receive data

    # Complex Numbers are converted to real Numbers:
    R_real_value = np.hstack((np.real(Rece_data), np.imag(Rece_data)))
    CSI_real_value = (np.hstack((np.real(CSI), np.imag(CSI))))
    L_real_value = (np.hstack((np.real(L_1), np.imag(L_1))))


    loss_H = tf.reduce_mean(tf.square(target_H - H_1))/tf.reduce_mean(tf.square(target_H))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        feed_dict = {xs: R_real_value, target_H: CSI_real_value, target_D: L_real_value}
        saver.restore(sess, file_name)
        # out_data = data_map(sess.run(D_1, feed_dict=feed_dict))
        # ber = BER(out_data, data_map(sess.run(target_D, feed_dict=feed_dict)))

        print('*' * 50)
        print("MSE-CSI：")
        print(sess.run(loss_H, feed_dict=feed_dict))
        # print("BER-data：")
        # print(ber)
        # print('*' * 50)
