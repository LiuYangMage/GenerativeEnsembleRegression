#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy import stats

import argparse
# In[2]:


parser = argparse.ArgumentParser(description='GAN-SODE')
parser.add_argument('--GPU', type=int, default=0, help='GPU ID')
parser.add_argument('-prb', '--problem', choices=['1e4', '1e5', '1e6', '1e7', 'Inf'])
parser.add_argument('-trs', '--train_size', type=int, default=10000)
parser.add_argument('-dim', '--dim', type=int, default=1)
parser.add_argument('-its', '--iterations', type=int, default=100000)
parser.add_argument('-res', '--restore', type=int, default=-1)
parser.add_argument('--seed',type=int, default=0, help='random seed')
parser.add_argument('--lasso', type=float, default = 0.0, help='use L1 penalty on the terms, not for nn')
# parser.add_argument('--GAN',help='version of GAN')
parser.add_argument('--grad', action= 'store_true')
parser.add_argument('--drift', choices=['2term', '4term', 'nn'], help='the format of the drift')
parser.add_argument('--float64', action= 'store_true')
parser.add_argument('--diff', choices=['known','const'], default='known')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--bs', type=int, default= 1000)


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
steps = [20, 50, 100]
ref_steps = [0, 500]

total_steps = 500
frames = len(steps)
ref_frames = len(ref_steps)


ref = {i: np.load('data1D/ref_{}.npz'.format(i))['ref'] for i in ref_steps + steps}
Qdata = [ref[A][np.random.choice(len(ref[A]),args.train_size,False),:] for A in steps]

# In[3]:



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
    if args.diff == 'known':
        diff =  1
    elif args.diff == 'const':
        diff =  tf.nn.softplus(s_W[0])
    else:
        raise NotImplementedError
    return diff


def fun_drift(x):
    if args.drift == '2term':
        drift = d_W[0] * x + d_W[1] * x**3
    elif args.drift == '4term':
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

        u[i+1] = u[i] + dt * drift +  1 * np.sqrt(dt) * diff * tf.random.normal([bs, dim], mean=0.0, stddev=1.0, dtype = dtype)
        print(i+1, end = ' ', flush = True)
    
    return u[-1], u



def mkfigure_train_1D(title):

    plt.figure(figsize=(10,6 * frames))
    plotid = 0
    for plotid in range(frames):
        s = steps[plotid]
        plt.subplot(frames,1,plotid + 1)
        init = np.concatenate([sess.run(Gs[s]) for i in range(10)], axis = 0)
        sns.kdeplot(init[:,0], label = '10,000 \n generated sample')
        sns.kdeplot(Qdata[plotid][:,0], label = '{} \n training samples'.format(len(Qdata[plotid])))         
        sns.kdeplot(ref[s][np.random.choice(len(ref[s]),10000,False),0], label = '10,000 \n  MC samples')
        plt.title('t = {}'.format(s/100))
        plt.legend()
        plt.xlim(-5,5)
    
    plt.savefig(savedir+ '/' + title + '.eps', format = 'eps')

def mkfigure_ref_1D(title):
    plt.figure(figsize=(10, 6 * ref_frames))
    plotid = 0
    for plotid in range(ref_frames):
        s = ref_steps[plotid]
        plt.subplot(ref_frames,1,plotid + 1)
        init = np.concatenate([sess.run(Gs[s]) for i in range(10)], axis = 0)
        sns.kdeplot(init[:,0], label = '10,000 \n generated sample')
        sns.kdeplot(ref[s][np.random.choice(len(ref[s]),10000,False),0], label = '10,000 \n  MC samples')
        plt.title('t = {}'.format(s/100))
        plt.legend()
        plt.xlim(-5,5)

    plt.savefig(savedir+ '/' + title + '.eps', format = 'eps')

def save_sample(title, steps, repeat = 100):
    init = []
    for s in steps:
        init.append(np.concatenate([sess.run(Gs[s]) for i in range(repeat)], axis = 0))
    np.savez(savedir + '/' + title + '.npz', steps = np.array(steps), Gdata = np.array(init))

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


def mkfigure_drift(title, dim1, dim2, sc = 0.1):

    plt.figure(figsize=(10,10))
    current_drift_x, current_drift_ref, current_drift = sess.run([vis_drift_x, vis_drift_ref, vis_drift])

    for index in range(len(current_drift_x)):
        plt.arrow(current_drift_x[index,dim1], current_drift_x[index,dim2], sc * current_drift[index, dim1], sc * current_drift[index,dim2],
                  head_width=0.02,
                  head_length=0.02, color = 'r')
        plt.arrow(current_drift_x[index,dim1], current_drift_x[index,dim2], sc * current_drift_ref[index, dim1], sc * current_drift_ref[index,dim2],
            head_width=0.02,
            head_length=0.02, color = 'k')
    
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.savefig(savedir+ '/' + title + '.eps', format = 'eps')




layer_dims = [zdim] + 3*[128] + [dim] 
L = len(layer_dims)
G_W = [tf.get_variable('G_W_{}'.format(l), [layer_dims[l-1], layer_dims[l]], dtype=dtype, initializer=tf.contrib.layers.xavier_initializer()) for l in range(1, L)]
G_b = [tf.get_variable('G_b_{}'.format(l), [1,layer_dims[l]], dtype=dtype, initializer=tf.zeros_initializer()) for l in range(1, L)]

if args.diff == 'known':
    s_W = []
    s_b = []
elif args.diff == 'const':
    s_W = [tf.Variable(np.zeros((1,dim)), dtype = dtype)]
    s_b = []
else :
    raise NotImplementedError

if args.drift == '2term':
    d_W = [tf.Variable(np.zeros((1,dim)), dtype = dtype) for i in range(2)]
    d_b = []
elif args.drift == '4term':
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

# ts = [i*dt*tf.ones([bs,1]) for i in steps]

Zs = tf.random.normal([bs, zdim], 0, 1, dtype=dtype)
Is = initgenerator(Zs, G_W, G_b)

_, Gs = generator(Is, total_steps, dt, bs)

Qs = [tf.placeholder(dtype, [bs,dim]) for i in range(frames)]


# In[10]:


num_projections = 1000
loss_PQ = [None for i in range(frames)]

for i in range(frames):

    theta = tf.nn.l2_normalize(tf.random_normal(shape=[dim, num_projections], dtype = dtype), axis=0)
    projected_true = tf.transpose(tf.matmul(Qs[i], theta))
    projected_fake = tf.transpose(tf.matmul(Gs[steps[i]], theta))


    sorted_true, true_indices = tf.nn.top_k(projected_true,bs)
    sorted_fake, fake_indices = tf.nn.top_k(projected_fake,bs)

    loss_PQ[i] = tf.reduce_mean(tf.square(sorted_true - sorted_fake))
    print(i, end = ' ', flush = True)


loss_PQ_all = tf.reduce_sum(loss_PQ)  
if args.lasso > 0:
    loss_PQ_all = loss_PQ_all + args.lasso * tf.reduce_sum([tf.abs(i) for i in d_W])
G_op = tf.train.AdamOptimizer(learning_rate = args.lr).minimize(loss_PQ_all, var_list = G_W + G_b + d_W + d_b + s_W + s_b)    


drift_x = np.linspace(-3,3,301)[:,None]
vis_drift_x = tf.constant(drift_x, dtype = dtype)
vis_drift_ref = vis_drift_x - vis_drift_x ** 3
vis_drift = fun_drift(vis_drift_x)



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


# In[26]:


savedir = 'save{}D-drift{}-diff{}-total{}-trainsize{}-float64{}-grad{}-seed{}'.format(
    args.dim, args.drift, args.diff, args.problem, args.train_size, args.float64, args.grad, args.seed)

 
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

        mkfigure_train_1D('train{}'.format(it))
        mkfigure_ref_1D('ref{}'.format(it))
        mkfigure_drift_1D('drift{}'.format(it))
 
    sess.run(G_op, feed_dict= {Qs[t]: Qdata[t][np.random.choice(len(Qdata[t]), bs), :] for t in range(frames)})
    it += 1
    print('.', end = '', flush = True)


save_sample('samples', steps + ref_steps, 1000)
