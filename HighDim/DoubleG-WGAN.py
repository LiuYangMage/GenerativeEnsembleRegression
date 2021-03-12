#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
# import tensorflow_probability as tfp
from scipy import stats
from utils import *

import argparse
# In[2]:


parser = argparse.ArgumentParser(description='GAN-SODE')
parser.add_argument('--gpu', type=int, default=-1, help='GPU ID')
parser.add_argument('--dim', type=int)
parser.add_argument('--trs', type=int, default = 100000)
parser.add_argument('--its', type=int, default=100000)
parser.add_argument('--bs', type=int, default= 1000)
parser.add_argument('--res', type=int, default=-1)
parser.add_argument('--lasso', type=float, default = 0.0, help='use L1 penalty on the terms, not for nn')
parser.add_argument('--drift', choices=['4term'], default='4term')
parser.add_argument('--diff', choices=['const'], default='const')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--float64', action= 'store_true')
parser.add_argument('--frames', type=int, default=3)
parser.add_argument('--gradclip', action= 'store_true')
parser.add_argument('--genact', type=str, default='tanh')
parser.add_argument('--diffact', type=str, default='softplus')
parser.add_argument('--lamda', type=float, default=0.1)
parser.add_argument('--dgratio', type=int, default=5)
parser.add_argument('--gentype', type=str, default='net')
parser.add_argument('--distype', type=str, default='resnet')
parser.add_argument('--Dsize', type=str, default='3x128')
parser.add_argument('--preprc', action= 'store_true')
parser.add_argument('--seed',type=int, default=0, help='random seed')


args = parser.parse_args()

if args.gpu > -1:
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'   # see issue #152
    os.environ['CUDA_VISIBLE_DEVICES']= str(args.gpu)


savedir = 'save'
for key,value in vars(args).items():
    print(key, value)
    if key in ['gpu', 'its', 'trs', 'res', 'lasso', 'drift', 'diff']:
        pass
    else:
        savedir = savedir + '-' + str(key)+str(value)

print(savedir)


bs = args.bs
seed = args.seed
lamda = args.lamda

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
if args.frames == 3:
    steps = [20, 50, 100]
elif args.frames == 6:
    steps = [10, 20, 50, 70, 100, 150]
ref_steps = [0, 500]

total_steps = 500
frames = len(steps)
ref_frames = len(ref_steps)

ref = {i: np.load('../Low2HighCoupleDiffWGAN/data{}D/ref_{}.npz'.format(dim,i))['ref'] for i in ref_steps + steps}
Qdata = [ref[A][np.random.choice(len(ref[A]),args.trs, False),:] for A in steps]


def diffact_np(x):
    if args.diffact == 'softplus':
        return np.log(1+np.exp(x))
    elif args.diffact == 'square':
        return x**2
    else:
        raise NotImplementedError


def fun_diff(x):
    if args.diff == 'const':
        if args.diffact == 'softplus':
            diff_0 = tf.nn.softplus(s_W[0])
        elif args.diffact == 'square':
            diff_0 = s_W[0]**2
        else:
            raise NotImplementedError
        diff_1 = s_W[1][:,:-1]
    else:
        raise NotImplementedError
    return diff_0, diff_1

def fun_drift(x):
    if args.drift == '4term':
        drift = d_W[0] + d_W[1] * x + d_W[2] * x**2 + d_W[3] * x**3
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
        diff_0, diff_1 = fun_diff(u[i])
        rand = tf.random.normal([bs, dim], mean=0.0, stddev=1.0, dtype = dtype)
        u[i+1] = u[i] + dt * drift + np.sqrt(dt) * diff_0 * rand  \
                    + tf.concat([tf.zeros((bs, 1), dtype = dtype), np.sqrt(dt) * diff_1 * rand[:,:-1]], axis = 1)
        print(i+1, end = ' ', flush = True)
    
    return u[-1], u


def save_sample(title, save_steps, repeat = 100):
    init = []
    for s in save_steps:
        if args.gentype == 'data':
            init.append(np.concatenate([sess.run(Gs[s-steps[0]], 
            feed_dict= {Qs[t]: Qdata[t][np.random.choice(len(Qdata[t]), bs), :] for t in [0]})
             for i in range(repeat)], axis = 0))
        else:
            init.append(np.concatenate([sess.run(Gs[s]) for i in range(repeat)], axis = 0))
    np.savez(savedir + '/' + title + '.npz', steps = np.array(save_steps), Gdata = np.array(init))


def gradient_pernalty(G, T, f, batch_size = bs):
    zu = tf.random_uniform([batch_size,1], minval=0, maxval=1, dtype=dtype)
    D_interpolates = zu * T + (1 - zu) * G  
    D_disc_interpolates = f(D_interpolates)
    D_gradients = tf.gradients(D_disc_interpolates, [D_interpolates])[0]
    D_slopes = tf.norm(D_gradients, axis = 1)
    D_gradient_penalty = tf.reduce_mean((D_slopes-1)**2)
    return D_gradient_penalty

def maxtensor(x):
    xnp = sess.run(x)
    return np.max([np.max(np.abs(i)) for i in xnp])

s_W = [tf.Variable(np.zeros((1,dim)), dtype = dtype) for i in range(2)]
d_W = [tf.Variable(np.zeros((1,dim))-0.5, dtype = dtype) for i in range(4)]

Qs = [tf.placeholder(dtype, [bs,dim]) for i in range(frames)]

Zs = tf.random.normal([bs, zdim], 0, 1, dtype=dtype)


if args.genact == 'swish':
    genact = tf.nn.swish
elif args.genact == 'tanh':
    genact = tf.nn.tanh
elif args.genact == 'relu':
    genact = tf.nn.relu
elif args.genact == 'lrelu':
    genact = tf.nn.leaky_relu
elif args.genact == 'softplus':
    genact = tf.nn.softplus
else:
    raise NotImplementedError

if args.gentype == 'data':
    Is = Qs[0]
    G_vars = d_W + s_W
elif args.gentype == 'net':
    initG = NeuralNet([zdim] + 3*[128] + [dim], name = 'G', dtype = dtype, act = genact)
    Is = initG(Zs)
    G_vars = initG.trainable_variables() + d_W + s_W
elif args.gentype == 'ode':
    initG = ODENet(NeuralNet([dim+1] + 3*[128] + [dim], name = 'G', dtype = dtype, act = genact))
    Is = initG(Zs)
    G_vars = initG.trainable_variables() + d_W + s_W
else:
    raise NotImplementedError

_, Gs = generator(Is, total_steps, dt, bs)

loss_PQ = []
loss_D = []
D_vars = []
ds = []

start = 1 if args.gentype == 'data' else 0
for i in range(start, frames):
    depth, width = args.Dsize.split('x')
    layer_dims = [dim] + int(depth)*[int(width)] + [1] 
    if args.distype == 'net':
        ds.append(NeuralNet(layer_dims, name = 'D_{}'.format(i), dtype = dtype, act = tf.nn.leaky_relu))
    elif args.distype == 'resnet':
        ds.append(ResNet(layer_dims, name = 'D_{}'.format(i), dtype = dtype, act = tf.nn.leaky_relu))
    else:
        raise NotImplementedError

    D_vars = D_vars + ds[-1].trainable_variables()
    if args.gentype == 'data':
        thisT = Qs[i]; thisG = Gs[steps[i] - steps[0]]
    else:
        thisT = Qs[i]; thisG = Gs[steps[i]]

    if args.preprc:
        thisT = 2*tf.tanh(thisT*0.5)
        thisG = 2*tf.tanh(thisG*0.5)

    surrogate_loss = tf.reduce_mean(ds[-1](thisT) - ds[-1](thisG))
    loss_PQ.append(surrogate_loss)
    loss_D.append(- surrogate_loss + lamda *  gradient_pernalty(thisG, thisT, ds[-1], batch_size = bs))

    print(i, end = ' ', flush = True)

loss_PQ_all = tf.reduce_sum(loss_PQ)
loss_D_all = tf.reduce_sum(loss_D)


if args.gradclip:
    G_op_original = tf.train.AdamOptimizer(learning_rate = args.lr, beta1=0.5, beta2=0.9)
    G_op = tf.contrib.estimator.clip_gradients_by_norm(G_op_original, clip_norm=0.5).minimize(loss_PQ_all, var_list = G_vars)

    D_op_original = tf.train.AdamOptimizer(learning_rate = args.lr, beta1=0.5, beta2=0.9)
    D_op = tf.contrib.estimator.clip_gradients_by_norm(D_op_original, clip_norm=0.5).minimize(loss_D_all, var_list = D_vars)
else:
    G_op = tf.train.AdamOptimizer(learning_rate = args.lr, beta1=0.5, beta2=0.9).minimize(loss_PQ_all, var_list = G_vars)
    D_op = tf.train.AdamOptimizer(learning_rate = args.lr, beta1=0.5, beta2=0.9).minimize(loss_D_all, var_list = D_vars)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


# In[26]:

if not os.path.exists(savedir):
    os.mkdir(savedir)
saver = tf.train.Saver(max_to_keep=1000)

with open(savedir + '/args.txt', 'w') as f:
    print(args, file=f)

os.system('cp *.py ./{}'.format(savedir))

if args.res >= 0:
    it = args.res
    saver.restore(sess, savedir+'/' + str(it) + '.ckpt')
    drift_history = [np.array(A) for A in np.load(savedir+'/drift_history.npz')['drift_history']][:-1]
    diff_history = [np.array(A) for A in np.load(savedir+'/diff_history.npz')['diff_history']][:-1]

else:
    np.savez(savedir + '/train.npz', Qdata = np.array(Qdata), steps = np.array(steps))
    it = 0

    drift_history = []
    diff_history = []

for _ in range(args.its - it + 1):

    if it % 50000 == 0:
        save_path = saver.save(sess, savedir+'/' + str(it) + '.ckpt')

    if it % 500 ==0:
        print(it, flush = True)
        drift_history.append(sess.run(d_W))
        diff_history.append(sess.run(s_W))
        print(drift_history[-1], diffact_np(diff_history[-1][0]), diff_history[-1][1])
        np.savez(savedir+'/drift_history.npz', drift_history = np.array(drift_history))
        np.savez(savedir+'/diff_history.npz', diff_history = np.array(diff_history))
    
    for _ in range(args.dgratio):
        sess.run(D_op, feed_dict= {Qs[t]: Qdata[t][np.random.choice(len(Qdata[t]), bs), :] for t in range(frames)})
    sess.run(G_op, feed_dict= {Qs[t]: Qdata[t][np.random.choice(len(Qdata[t]), bs), :] for t in range(frames)})
    # add this for 50D
    # print(it, sess.run(loss_PQ + loss_D, feed_dict= {Qs[t]: Qdata[t][np.random.choice(len(Qdata[t]), bs), :] for t in range(frames)}))
    # print(maxtensor(G_vars), maxtensor(D_vars))

    it += 1
    print('.', end = '', flush = True)

save_sample('samples', steps + ref_steps)
