#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats

from nets import *


import argparse
# In[2]:


parser = argparse.ArgumentParser(description='GAN-SODE')
parser.add_argument('--GPU', type=int, default=0, help='GPU ID')
parser.add_argument('-dim', '--dim', type = int, default= 2)
parser.add_argument('--metric', choices=['SW','WGAN-GP'], default='SW')
parser.add_argument('-alp', '--alp', type=float, default=0.5)
parser.add_argument('-dt','--dt', type=float, default=0.01)
parser.add_argument('-its', '--iterations', type=int, default=200000)
parser.add_argument('--msbs', type=int, default= 9976) # 1024 for 1D
parser.add_argument('--msts', type=int, default= 0)
parser.add_argument('-res', '--restore', type=int, default=-1)
parser.add_argument('--seed',type=int, default=0, help='random seed')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--clip', type=float, default=1e-2, help = 'clip the distance between particles')
parser.add_argument('--float64', action= 'store_true')
parser.add_argument('--snapnum', type=int, default=16)
parser.add_argument('--weight', type=float, default=1.0)
parser.add_argument('--umap', choices=['none', 'nml'], default= 'nml')
parser.add_argument('--vmap', choices=['none', 'nml'], default= 'none')
parser.add_argument('--phase', choices=['u', 'uv'], default= 'u')
parser.add_argument('--ntsample', default='independent')
parser.add_argument('--ntts', type=int, default=10)
parser.add_argument('--ntbs', type=int, default=16)
parser.add_argument('--ntra', type=float, default=0.2)
parser.add_argument('--ntns', type=int, default=10000) # should be reduced if initgen == 'data'
parser.add_argument('--initgen', choices=['fnn','data'], default='fnn')
parser.add_argument('--alpscale', type=float, default=2)



args = parser.parse_args()

print('-------------------------------------------')
print(args)
print('-------------------------------------------')


if args.GPU >= 0:
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'   # see issue #152
    os.environ['CUDA_VISIBLE_DEVICES']= str(args.GPU)
else:
    pass


seed = args.seed
tf.reset_default_graph()
tf.set_random_seed(seed)
np.random.seed(seed)

if args.float64:
    dtype = tf.float64
else:
    dtype = tf.float32

if args.snapnum == 16:
    snaps = [i*10 for i in range(5,21)]
else:
    raise NotImplementedError

if args.initgen == 'fnn':
    pass
elif args.initgen == 'data':
    snaps = [i-50 for i in snaps]
else:
    raise NotImplementedError


ref_snaps = [i*10 for i in range(5)]
max_step = int(np.max(ref_snaps+snaps))

if args.dim == 2:
    if args.alp == 0.5 and args.clip == 1e-2:
        refu = np.load('2Ddata-05-s{}/ref_initrand_1e-2.npz'.format(seed))['x']
        refv = np.load('2Ddata-05-s{}/ref_initrand_1e-2.npz'.format(seed))['v']
    elif args.alp == 0.5 and args.clip == 4e-2:
        refu = np.load('2Ddata-05-s{}/ref_initrand_4e-2.npz'.format(seed))['x']
        refv = np.load('2Ddata-05-s{}/ref_initrand_4e-2.npz'.format(seed))['v']
    elif args.alp == 0.5 and args.clip == 8e-2:
        refu = np.load('2Ddata-05-s{}/ref_initrand_8e-2.npz'.format(seed))['x']
        refv = np.load('2Ddata-05-s{}/ref_initrand_8e-2.npz'.format(seed))['v']
    elif args.alp == 0.5 and args.clip == 10e-2:
        refu = np.load('2Ddata-05-s{}/ref_initrand_10e-2.npz'.format(seed))['x']
        refv = np.load('2Ddata-05-s{}/ref_initrand_10e-2.npz'.format(seed))['v']
    else:
        raise NotImplementedError
elif args.dim == 1:
    if args.alp == 0.5 and args.clip == 1e-2:
        refu = np.load('1Ddata-05-s{}/ref_initrand.npz'.format(seed))['x']
        refv = np.load('1Ddata-05-s{}/ref_initrand.npz'.format(seed))['v']
    else:
        raise NotImplementedError
else:
    raise NotImplementedError


if args.initgen == 'fnn':
    pass
elif args.initgen == 'data':
    refu = refu[50:,...]
    refv = refv[50:,...]
else:
    raise NotImplementedError


udata = [refu[s,:] for s in snaps]
vdata = [refv[s,:] for s in snaps]

msu = [tf.placeholder(dtype, [args.msbs, args.dim]) for i in range(len(snaps))]
msv = [tf.placeholder(dtype, [args.msbs, args.dim]) for i in range(len(snaps))]


nt_snaprnd = tf.placeholder(tf.int32, [args.ntts])
ms_snaprnd = tf.placeholder(tf.int32, [args.msts])

if args.initgen == 'fnn':
    initgenerator = InitGenerator_FNN(args.dim, dtype)
elif args.initgen == 'data':
    initgenerator = InitGenerator_data(args.dim, dtype, udata[0])
else:
    raise NotImplementedError

odegenerator = ODEGenerator(args.dim, dtype, max_step, args.dt)


alpha_raw = tf.Variable(0.0, dtype=dtype)
alpha = tf.nn.sigmoid(alpha_raw)*args.alpscale

newton = Newton(args.dim, dtype, ntsample = args.ntsample,
            bs = args.ntbs, snap_bs = args.ntts, ns = args.ntns, radius = args.ntra,
            clip = args.clip, alpha = alpha)


measuremetric = MeasureMetric(args.dim, dtype, metric = args.metric,
            phase = args.phase, uref = udata, vref = vdata, snapids = snaps, 
            umap = args.umap, vmap = args.vmap, bs = args.msbs, snap_bs = args.msts)


newton_loss = newton.get_loss(initgenerator, odegenerator, nt_snaprnd)

measuremetric_loss = measuremetric.get_loss(initgenerator, odegenerator, msu, msv, ms_snaprnd)


g_loss = measuremetric_loss + newton_loss * args.weight

g_op = tf.train.AdamOptimizer(learning_rate = args.lr).minimize(g_loss, 
        var_list = initgenerator.trainable_variables() + odegenerator.trainable_variables() + [alpha_raw])


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


# In[26]:

savedir = 'save2D'

for key,value in vars(args).items():
    print(key, value)
    if key in ['GPU']:
        pass
    elif key in []:
        savedir = savedir + '-' + str(key)
        for val in value:
            savedir = savedir + '+' + str(val)
    else:
        savedir = savedir + '-' + str(key)+str(value)

if not os.path.exists(savedir):
    os.mkdir(savedir)
saver = tf.train.Saver(max_to_keep=1000)


with open(savedir + '/args.txt', 'w') as f:
    print(args, file=f)

os.system('cp *.py ./{}'.format(savedir))

if args.restore >= 0:
    it = args.restore
    saver.restore(sess, savedir+'/' + str(it) + '.ckpt')
    alpha_history = [np.array(A) for A in np.load(savedir+'/alpha_history.npz')['alpha_history']][:it//500]
    loss_history = [np.array(A) for A in np.load(savedir+'/loss_history.npz')['loss_history']][:it//500]

else:
    np.savez(savedir + '/train.npz', udata = np.array(udata), vdata = np.array(vdata), snaps = np.array(snaps))
    it = 0
    alpha_history = []
    loss_history = []


for _ in range(args.iterations - it + 1):

    if it % 10000 == 0:
        save_path = saver.save(sess, savedir+'/' + str(it) + '.ckpt')

    if it % 500 ==0:

        alpha_history.append(sess.run(alpha))
        np.savez(savedir+'/alpha_history.npz', alpha_history = np.array(alpha_history))

        if args.phase == 'uu':
            indice = np.random.choice(len(udata[0]), args.msbs, replace = False) 
            msindices = [indice for t in range(len(snaps))]
        else:
            msindices = [np.random.choice(len(udata[t]), args.msbs, replace = False) for t in range(len(snaps))]
        
        ntindices = [np.random.choice(len(udata[t]), args.ntbs, replace = True) for t in range(len(snaps))]
        nbindices = [np.random.choice(len(udata[t]), args.ntns, replace = True) for t in range(len(snaps))]
        
        loss_history.append(sess.run(measuremetric.losses, 
                feed_dict = {**{msu[t]: udata[t][msindices[t], :] for t in range(len(snaps))},
                             **{msv[t]: vdata[t][msindices[t], :] for t in range(len(snaps))},
                             **{nt_snaprnd: np.random.choice(max_step+1, args.ntts, replace = False)},
                             **{ms_snaprnd: np.random.choice(len(snaps), args.msts, replace = False)}}))
        np.savez(savedir+'/loss_history.npz', loss_history = np.array(loss_history))
        
        print('\n', it, end = ' ', flush = True)
        print(alpha_history[-1], end = ' ', flush = True)
        print(loss_history[-1], end = '\n', flush = True)

    if args.phase == 'uu':
        indice = np.random.choice(len(udata[0]), args.msbs, replace = False) 
        msindices = [indice for t in range(len(snaps))]
    else:
        msindices = [np.random.choice(len(udata[t]), args.msbs, replace = False) for t in range(len(snaps))]
    
    ntindices = [np.random.choice(len(udata[t]), args.ntbs, replace = True) for t in range(len(snaps))]
    nbindices = [np.random.choice(len(udata[t]), args.ntns, replace = True) for t in range(len(snaps))]


    sess.run(g_op, feed_dict = {**{msu[t]: udata[t][msindices[t], :] for t in range(len(snaps))},
                                **{msv[t]: vdata[t][msindices[t], :] for t in range(len(snaps))},
                                **{nt_snaprnd: np.random.choice(max_step+1, args.ntts, replace = False)},
                                **{ms_snaprnd: np.random.choice(len(snaps), args.msts, replace = False)}})


    it += 1
    print('.', end = '', flush = True)

