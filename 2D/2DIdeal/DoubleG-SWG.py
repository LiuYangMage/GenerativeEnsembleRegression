#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from scipy import stats

import argparse
# In[2]:


parser = argparse.ArgumentParser(description='GAN-SODE')
parser.add_argument('--GPU', type=int, default=0, help='GPU ID')
parser.add_argument('-dim', '--dim', type = int, default= 2)
parser.add_argument('--GAN', choices=['SW','WGAN-GP'], default='WGAN-GP')
parser.add_argument('-trs', '--train_size', type=int, default= 100000)
parser.add_argument('-its', '--iterations', type=int, default=200000)
parser.add_argument('--bs', type=int, default= 1000)
parser.add_argument('-res', '--restore', type=int, default=-1)
parser.add_argument('--seed',type=int, default=0, help='random seed')
parser.add_argument('--lasso', type=float, default = 0.0, help='use L1 penalty on the terms, not for nn')
parser.add_argument('--drift', choices=['4term', 'nn'], default='4term')
parser.add_argument('--diff', choices=['const'], default='const')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--float64', action= 'store_true')
parser.add_argument('--grad', action= 'store_true')
parser.add_argument('--frames', type=int, default=3)


args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES']= str(args.GPU)


bs = args.bs
seed = args.seed
lamda = 0.1

tf.reset_default_graph()
tf.set_random_seed(seed)
np.random.seed(seed)

if args.float64:
    dtype = tf.float64
else:
    dtype = tf.float32




dim = args.dim
zdim = args.dim
dt = 0.01
if args.frames == 7:
    steps = [0, 10, 20, 30, 50, 70, 100]
elif args.frames == 3:
    steps = [20, 50, 100]
ref_steps = []

total_steps = 100
frames = len(steps)
ref_frames = len(ref_steps)


ref = {i: np.load('data{}D-s{}/ref_{}.npz'.format(dim,seed,i))['ref'] for i in ref_steps + steps}
Qdata = [ref[A][np.random.choice(len(ref[A]),args.train_size, False),:] for A in steps]


def feed_NN(X, W, b, act = tf.nn.tanh):
    A = X
    L = len(W)
    for i in range(L-1):
        A = act(tf.add(tf.matmul(A, W[i]), b[i]))  
    return tf.add(tf.matmul(A, W[-1]), b[-1])

def initgenerator(X, W, b):
    y = feed_NN(X,W,b, act= tf.nn.tanh)
    return y


# In[4]:

def fun_diff(x):
    if args.diff == 'const':
        diff_raw =  tf.concat([tf.nn.softplus(s_W[0]), s_W[1], 
                          tf.zeros((1,), dtype = dtype), tf.nn.softplus(s_W[2])], axis = 0)
        diff = tf.reshape(diff_raw, [2,2])
    else:
        raise NotImplementedError
    return diff


def fun_drift(x):
    if args.drift == '4term':
        po = -(x[:,0] + d_W[0])**2 * (x[:,1] + d_W[1])**2 \
             -(x[:,0] + d_W[2])**2 * (x[:,1] + d_W[3])**2
        drift = tf.gradients(po, x)[0]

    elif args.drift == 'nn':
        po = feed_NN(x, d_W, d_b, act= tf.nn.tanh)
        drift = tf.gradients(po, x)[0]
    else:
        raise NotImplementedError
    return drift


def generator(x, steps, dt, bs = bs):
    '''
    x shape: [bs, dim]
    '''
    u = [None for i in range(steps + 1)]
    u[0] = x
    print(0, end = ' ', flush = True)
    
    for i in range(steps):
        drift = fun_drift(u[i])
        diff = fun_diff(u[i])

        u[i+1] = u[i] + dt * drift +  np.sqrt(dt) * tf.matmul(tf.random.normal([bs, dim], mean=0.0, stddev=1.0, dtype = dtype), diff)
        print(i+1, end = ' ', flush = True)
    
    return u[-1], u


def save_drift(title, dim1, dim2, sc = 0.1):

    current_drift_x, current_drift_ref, current_drift = sess.run([vis_drift_x, vis_drift_ref, vis_drift])

    np.savez(savedir + '/' + title + '.npz', x = current_drift_x, 
                                            drift = current_drift, 
                                            drift_ref = current_drift_ref)



def save_sample(title, steps, repeat = 100):
    init = []
    for s in steps:
        init.append(np.concatenate([sess.run(Gs[s]) for i in range(repeat)], axis = 0))
    np.savez(savedir + '/' + title + '.npz', steps = np.array(steps), Gdata = np.array(init))


class Net(object):
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def __call__(self, x):
        return feed_NN(x, self.W, self.b, act= tf.nn.leaky_relu)


def gradient_pernalty(G, T, f, batch_size = bs):
    zu = tf.random_uniform([batch_size,1], minval=0, maxval=1, dtype=dtype)
    D_interpolates = zu * T + (1 - zu) * G  
    D_disc_interpolates = f(D_interpolates)
    D_gradients = tf.gradients(D_disc_interpolates, [D_interpolates])[0]
    D_slopes = tf.norm(D_gradients, axis = 1)
    D_gradient_penalty = tf.reduce_mean((D_slopes-1)**2)
    return D_gradient_penalty


layer_dims = [zdim] + 3*[128] + [dim] 
L = len(layer_dims)
G_W = [tf.get_variable('G_W_{}'.format(l), [layer_dims[l-1], layer_dims[l]], dtype=dtype, initializer=tf.contrib.layers.xavier_initializer()) for l in range(1, L)]
G_b = [tf.get_variable('G_b_{}'.format(l), [1,layer_dims[l]], dtype=dtype, initializer=tf.zeros_initializer()) for l in range(1, L)]




if args.drift == '4term':
    d_W = [tf.Variable(np.random.normal(0,1,(1,)), dtype = dtype) for i in range(4)]
    d_b = []
elif args.drift == 'nn':
    layer_dims = [dim] + 3*[128] + [1] 
    L = len(layer_dims)
    d_W = [tf.get_variable('d_W_{}'.format(l), [layer_dims[l-1], layer_dims[l]], dtype=dtype, initializer=tf.contrib.layers.xavier_initializer()) for l in range(1, L)]
    d_b = [tf.get_variable('d_b_{}'.format(l), [1,layer_dims[l]], dtype=dtype, initializer=tf.zeros_initializer()) for l in range(1, L)]

s_W = [tf.Variable(np.zeros((1,)), dtype = dtype) for i in range(3)]

Qs = [tf.placeholder(dtype, [bs,dim]) for i in range(frames)]

Zs = tf.random.normal([bs, zdim], 0, 1, dtype=dtype)
Is = initgenerator(Zs, G_W, G_b)

_, Gs = generator(Is, total_steps, dt, bs)


num_projections = 1000
loss_PQ = [None for i in range(frames)]

if args.GAN == 'SW':

    for i in range(frames):

        theta = tf.nn.l2_normalize(tf.random_normal(shape=[dim, num_projections], dtype = dtype), axis=0)
        projected_true = tf.transpose(tf.matmul(Qs[i], theta))
        projected_fake = tf.transpose(tf.matmul(Gs[steps[i]], theta))


        sorted_true, true_indices = tf.nn.top_k(projected_true,bs)
        sorted_fake, fake_indices = tf.nn.top_k(projected_fake,bs)

        loss_PQ[i] = tf.reduce_mean(tf.square(sorted_true - sorted_fake))
        print(i, end = ' ', flush = True)

    loss_PQ_all = tf.reduce_sum(loss_PQ)  
    G_op = tf.train.AdamOptimizer(learning_rate = args.lr).minimize(loss_PQ_all, var_list = G_W + G_b + d_W + d_b + s_W)

elif args.GAN == 'WGAN-GP':

    loss_D = [None for i in range(frames)]

    D_W = []
    D_b = []

    for i in range(frames):
        layer_dims = [dim] + 3*[128] + [1] 
        L = len(layer_dims)
        D_W.append([tf.get_variable("D_W_{}_{}".format(i,l), [layer_dims[l-1], layer_dims[l]], dtype=dtype, initializer=tf.contrib.layers.xavier_initializer()) for l in range(1, L)])
        D_b.append([tf.get_variable("D_b_{}_{}".format(i,l), [1,layer_dims[l]], dtype=dtype, initializer=tf.zeros_initializer()) for l in range(1, L)])

        d = Net(D_W[i], D_b[i])

        surrogate_loss = tf.reduce_mean(d(Qs[i]) - d(Gs[steps[i]]))
        loss_PQ[i] = surrogate_loss
        loss_D[i] = - surrogate_loss + lamda *  gradient_pernalty(Gs[steps[i]], Qs[i], d, batch_size = bs)

        print(i, end = ' ', flush = True)

    loss_PQ_all = tf.reduce_sum(loss_PQ)  
    G_op = tf.train.AdamOptimizer(learning_rate = args.lr, beta1=0.5, beta2=0.9).minimize(loss_PQ_all, var_list = G_W + G_b + d_W + d_b + s_W)

    loss_D_all = tf.reduce_sum(loss_D)
    D_vars = sum(D_W, []) + sum(D_b, [])
    D_op = tf.train.AdamOptimizer(learning_rate = args.lr, beta1=0.5, beta2=0.9).minimize(loss_D_all, var_list = D_vars)


drift_x = np.linspace(-2,2,101)
drift_x2, drift_x1 = np.meshgrid(drift_x, drift_x)
drift_x = np.concatenate([drift_x1.reshape(-1,1), drift_x2.reshape(-1,1)], axis = 1)

vis_drift_x = tf.constant(drift_x, dtype = dtype)
vis_drift_ref = vis_drift_x - vis_drift_x ** 3
vis_drift = fun_drift(vis_drift_x)



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


# In[26]:


savedir = 'save-GAN{}-drift{}-diff{}-frames{}-float64{}-seed{}'.format(
        args.GAN, args.drift, args.diff, frames, args.float64, args.seed)

 
if not os.path.exists(savedir):
    os.mkdir(savedir)
saver = tf.train.Saver(max_to_keep=1000)


if args.restore >= 0:
    it = args.restore
    saver.restore(sess, savedir+'/' + str(it) + '.ckpt')
    diff_history = [np.array(A) for A in np.load(savedir+'/diff_history.npz')['diff_history']][:-1]
    if args.drift != 'nn':
        drift_history = [np.array(A) for A in np.load(savedir+'/drift_history.npz')['drift_history']][:-1]

else:
    np.savez(savedir + '/train.npz', Qdata = np.array(Qdata), steps = np.array(steps))
    it = 0

    diff_history = []
    if args.drift != 'nn':
        drift_history = []

for _ in range(args.iterations - it + 1):

    if it % 1000 == 0:
        save_path = saver.save(sess, savedir+'/' + str(it) + '.ckpt')
        save_drift('drift{}'.format(it), 0, 1)
    if it % 500 ==0:
        print(it, flush = True)
        if args.drift != 'nn':
            drift_history.append(sess.run(d_W))
            np.savez(savedir+'/drift_history.npz', drift_history = np.array(drift_history))
            print(drift_history[-1])
        diff_history.append(sess.run(s_W))
        np.savez(savedir+'/diff_history.npz', diff_history = np.array(diff_history))
        print(diff_history[-1])
    
    if args.GAN == 'WGAN-GP':
        for _ in range(5):
            sess.run(D_op, feed_dict= {Qs[t]: Qdata[t][np.random.choice(len(Qdata[t]), bs), :] for t in range(frames)})

    sess.run(G_op, feed_dict= {Qs[t]: Qdata[t][np.random.choice(len(Qdata[t]), bs), :] for t in range(frames)})
    it += 1
    print('.', end = '', flush = True)

save_sample('samples', steps + ref_steps)
