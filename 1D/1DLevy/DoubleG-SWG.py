#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats

import argparse
# In[2]:


parser = argparse.ArgumentParser(description='GAN-SODE')
parser.add_argument('--GPU', type=int, default=0, help='GPU ID')
parser.add_argument('-prb','--problem', choices=['origin','tanh'], default='origin')
parser.add_argument('-dim', '--dim', type = int, default=1)
parser.add_argument('-trs', '--train_size', type=int)
parser.add_argument('-its', '--iterations', type=int, default=100000)
parser.add_argument('--bs', type=int, default= 1000)
parser.add_argument('-res', '--restore', type=int, default=-1)
parser.add_argument('--seed',type=int, default=0, help='random seed')
parser.add_argument('--lasso', type=float, default = 0.0, help='use L1 penalty on the terms, not for nn')
parser.add_argument('--drift', choices=['4term', 'nn'], default='4term')
parser.add_argument('--diff', choices=['const'], default='const')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--float64', action= 'store_true')
parser.add_argument('--grad', action= 'store_true')


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
alpha = 1.5
steps = [20, 50, 100]
ref_steps = [0, 500]

total_steps = 500
frames = len(steps)
ref_frames = len(ref_steps)

ref = {i: np.load('data{}D/ref_{}.npz'.format(dim,i))['ref'] for i in ref_steps + steps}
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
        diff =  tf.nn.softplus(s_W[0])
    else:
        raise NotImplementedError
    return diff


def fun_drift(x):
    if args.drift == '4term':
        drift = d_W[0] + d_W[1] * x + d_W[2] * x**2 + d_W[3] * x**3
    elif args.drift == 'nn':
        drift = feed_NN(x, d_W, d_b, act= tf.nn.tanh)
        if args.grad:
            drift = tf.gradients(drift, x)[0]
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

        V = tf.random.uniform([bs, dim], -np.pi/2, np.pi/2, dtype = dtype)
        W = - tf.log(tf.random.uniform([bs, dim], dtype = dtype)) # 1 - uniform(0,1) = uniform(0,1)
        X = tf.sin(alpha * V) / tf.math.pow(tf.cos(V), 1/alpha) * tf.math.pow(tf.cos(V - alpha*V)/W, (1-alpha)/alpha)
        X_clip = tf.clip_by_value(X, -100, 100)
        update = dt * drift + dt ** (1/alpha) * diff * X_clip
        u[i+1] = u[i] + update
        print(i+1, end = ' ', flush = True)
    
    return u[-1], u


def mkfigure_1D(title, steplist):
    '''
    steplist: the list to plot
    '''
    plt.figure(figsize=(10,6 * len(steplist)))
    plotx = np.linspace(-5,5,401)

    for plotid in range(len(steplist)):
        s = steplist[plotid]
        plt.subplot(len(steplist),1,plotid + 1)

        init = np.concatenate([sess.run(Gs[s]) for i in range(10)], axis = 0)[:, 0]
        plt.plot(plotx, stats.gaussian_kde(init, 0.1)(plotx), lw=2,  label = '{} \n generated sample'.format(len(init)))

        try:
            init = Qdata[plotid][:,0]
            plt.plot(plotx, stats.gaussian_kde(init, 0.1)(plotx), lw=2,  label = '{} \n training samples'.format(len(init)))
        except:
            pass

        init = ref[s][np.random.choice(len(ref[s]),10000,False),0]
        plt.plot(plotx, stats.gaussian_kde(init, 0.1)(plotx), lw=2,  label = '{} \n MC samples'.format(len(init)))

        plt.title('t = {}'.format(s/100))
        plt.legend()
        plt.xlim(-5,5)
    
    plt.savefig(savedir+ '/' + title + '.eps', format = 'eps')


def mkfigure_drift_1D(title):

    plt.figure(figsize=(10,10))
    current_drift_x, current_drift_ref, current_drift = sess.run([vis_drift_x, vis_drift_ref, vis_drift])
    current_drift_x = current_drift_x[:,0]
    current_drift_ref = current_drift_ref[:,0]
    current_drift = current_drift[:,0]

    plt.plot(current_drift_x, current_drift, 'r-', label = 'inferred drift')
    plt.plot(current_drift_x, current_drift_ref, 'k-', label = 'exact drift')
    plt.legend()
    plt.xlim(-3,3)
    plt.ylim(min(current_drift) - 5, max(current_drift) + 5)
    
    np.savez(savedir + '/' + title + '.npz', x = current_drift_x, 
                                            drift = current_drift, 
                                            drift_ref = current_drift_ref)

    plt.savefig(savedir+ '/' + title + '.eps', format = 'eps')


def save_sample(title, steps, repeat = 100):
    init = []
    for s in steps:
        init.append(np.concatenate([sess.run(Gs[s]) for i in range(repeat)], axis = 0))
    np.savez(savedir + '/' + title + '.npz', steps = np.array(steps), Gdata = np.array(init))


layer_dims = [zdim] + 3*[128] + [dim] 
L = len(layer_dims)
G_W = [tf.get_variable('G_W_{}'.format(l), [layer_dims[l-1], layer_dims[l]], dtype=dtype, initializer=tf.contrib.layers.xavier_initializer()) for l in range(1, L)]
G_b = [tf.get_variable('G_b_{}'.format(l), [1,layer_dims[l]], dtype=dtype, initializer=tf.zeros_initializer()) for l in range(1, L)]

s_W = [tf.Variable(np.zeros((1,dim)), dtype = dtype)]
s_b = []

if args.drift == '4term':
    d_W = [tf.Variable(np.zeros((1,dim)), dtype = dtype) for i in range(4)]
    d_b = []
elif args.drift == 'nn':
    if args.grad:
        layer_dims = [dim] + 3*[128] + [1] 
    else:
        layer_dims = [dim] + 3*[128] + [dim]
    L = len(layer_dims)
    d_W = [tf.get_variable('d_W_{}'.format(l), [layer_dims[l-1], layer_dims[l]], dtype=dtype, initializer=tf.contrib.layers.xavier_initializer()) for l in range(1, L)]
    d_b = [tf.get_variable('d_b_{}'.format(l), [1,layer_dims[l]], dtype=dtype, initializer=tf.zeros_initializer()) for l in range(1, L)]

else:
    raise NotImplementedError



Qs = [tf.placeholder(dtype, [bs,dim]) for i in range(frames)]

Zs = tf.random.normal([bs, zdim], 0, 1, dtype=dtype)
Is = initgenerator(Zs, G_W, G_b)

_, Gs = generator(Is, total_steps, dt, bs)


num_projections = 1000
loss_PQ = [None for i in range(frames)]

for i in range(frames):

    theta = tf.nn.l2_normalize(tf.random_normal(shape=[dim, num_projections], dtype = dtype), axis=0)
    if args.problem == 'origin':
        projected_true = tf.transpose(tf.matmul(Qs[i], theta))
        projected_fake = tf.transpose(tf.matmul(Gs[steps[i]], theta))
    elif args.problem == 'tanh':
        projected_true = tf.transpose(tf.matmul(2 * tf.nn.tanh(Qs[i] * 0.5), theta))
        projected_fake = tf.transpose(tf.matmul(2 * tf.nn.tanh(Gs[steps[i]]* 0.5), theta))
    else:
        raise NotImplementedError

    sorted_true, true_indices = tf.nn.top_k(projected_true,bs)
    sorted_fake, fake_indices = tf.nn.top_k(projected_fake,bs)

    loss_PQ[i] = tf.reduce_mean(tf.square(sorted_true - sorted_fake))
    print(i, end = ' ', flush = True)


loss_PQ_all = tf.reduce_sum(loss_PQ)  
if args.drift != 'nn':
    loss_PQ_all = loss_PQ_all + args.lasso * tf.reduce_sum([tf.abs(i) for i in d_W])
else:
    loss_PQ_all = loss_PQ_all
G_op = tf.train.AdamOptimizer(learning_rate = args.lr).minimize(loss_PQ_all, var_list = G_W + G_b + d_W + d_b + s_W + s_b)


drift_x = np.linspace(-5,5,501)[:,None]
vis_drift_x = tf.constant(drift_x, dtype = dtype)
vis_drift_ref = vis_drift_x - vis_drift_x ** 3
vis_drift = fun_drift(vis_drift_x)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


# In[26]:

savedir = 'save{}D-{}-drift{}-diff{}-trainsize{}-float64{}-seed{}'.format(
        args.dim, args.problem, args.drift, args.diff, args.train_size, args.float64, args.seed)
 
if not os.path.exists(savedir):
    os.mkdir(savedir)
saver = tf.train.Saver(max_to_keep=1000)


if args.restore >= 0:
    it = args.restore
    saver.restore(sess, savedir+'/' + str(it) + '.ckpt')
    if args.drift != 'nn':
        drift_history = [np.array(A) for A in np.load(savedir+'/drift_history.npz')['drift_history']][:-1]
    if args.diff == 'const':
        diff_history = [np.array(A) for A in np.load(savedir+'/diff_history.npz')['diff_history']][:-1]


else:
    np.savez(savedir + '/train.npz', Qdata = np.array(Qdata), steps = np.array(steps))
    it = 0

    if args.drift != 'nn':
        drift_history = []
    if args.diff == 'const':
        diff_history = []

for _ in range(args.iterations - it + 1):

    if it % 1000 == 0:
        save_path = saver.save(sess, savedir+'/' + str(it) + '.ckpt')

    if it % 500 ==0:
        print(it, flush = True)

        if args.drift != 'nn':
            drift_history.append(sess.run(d_W))
            np.savez(savedir+'/drift_history.npz', drift_history = np.array(drift_history))
        if args.diff == 'const':
            diff_history.append(sess.run(s_W))
            np.savez(savedir+'/diff_history.npz', diff_history = np.array(diff_history))

    sess.run(G_op, feed_dict= {Qs[t]: Qdata[t][np.random.choice(len(Qdata[t]), bs), :] for t in range(frames)})
    it += 1
    print('.', end = '', flush = True)

save_sample('samples', steps + ref_steps, repeat=1000)
