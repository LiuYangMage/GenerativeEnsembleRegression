import numpy as np
import tensorflow as tf
from utils import *


class InitGenerator_FNN(object):
    def __init__(self, dim, dtype):

        layer_dims = [dim] + 3*[128] + [dim]
        self.net = Net(layer_dims, 'InitGenerator', dtype = dtype, act=tf.nn.tanh)

        self.dim = dim
        self.dtype = dtype
        print('InitGenerator Created')
    
    def __call__(self, bs, zstd = 1, return_den = False):
        return self.generate(bs, zstd)

    def trainable_variables(self):
        return self.net.trainable_variables()

    def generate(self, bs, zstd = 1):
        if not (isinstance(bs, list)): bs = [bs] 
        z = tf.random.normal(bs+[self.dim], 0, zstd, dtype = self.dtype)
        return self.net(z)


class InitGenerator_data(object):
    def __init__(self, dim, dtype, data):
        '''
        data: []
        '''
        self.dim = dim
        self.dtype = dtype
        self.data = tf.constant(data, dtype=dtype)
        self.datalen = len(data)
        print('InitGenerator Created')
    
    def __call__(self, bs, zstd = 1, return_den = False):
        return self.generate(bs, zstd)

    def trainable_variables(self):
        return []

    def generate(self, bs, zstd = 1):
        if not (isinstance(bs, list)): bs = [bs] 
        allbs = np.product(bs)
        if allbs == self.datalen:
            selected = self.data
        else:
            # random index without replacement
            ints = tf.constant(list(range(self.datalen)), dtype=tf.int64)[None,:]
            inx, _, _ = tf.random.uniform_candidate_sampler(
                ints,self.datalen, allbs, unique=True, range_max=self.datalen, seed=None, name=None)
            selected = tf.gather(self.data, inx, axis=0, batch_dims=0)
        thisdata = tf.reshape(selected,bs+[self.dim])
        return thisdata


class ODEGenerator(ODEFlow):
    def __init__(self, dim, dtype, steps, dt):
        super().__init__(dim, dtype, False, steps, dt, 'ODEGenerator')



class Newton(object):
    def __init__(self, dim, dtype, ntsample, bs, clip, alpha, ns, snap_bs, radius):
        self.dim = dim
        self.dtype = dtype
        self.bs = bs
        self.ns = ns
        self.snap_bs = snap_bs
        self.ntsample = ntsample
        self.radius = radius
        self.clip = clip
        self.alpha = alpha
        self.c = self.alpha * tf.exp(tf.math.lgamma((dim + self.alpha)/2)) / \
            (2 * np.pi** (self.alpha + dim/2) * tf.exp(tf.math.lgamma(1 - self.alpha/2)))


    def get_loss(self, initgenerator, odegenerator, snaprnd):
        if self.ntsample == 'independent':
            self.losses = self.get_loss_independent(initgenerator, odegenerator)
        else:
            raise NotImplementedError

        self.snaprnd = snaprnd
        if self.snap_bs > 0:
            self.loss_all = tf.reduce_mean(tf.gather(self.losses, self.snaprnd))
        else:
            self.loss_all = tf.reduce_mean(self.losses)
        return self.loss_all


    def get_loss_independent(self, initgenerator, odegenerator):
        inits = initgenerator(self.bs) #[bs, dim]
        fakeu, fakev, faket = odegenerator(self.bs, inits, return_full=True)
        snapnum = len(fakeu)

        # initsall = initgenerator([self.bs, self.ns]) #[bs, ns, dim]
        # neighu, neighv, _ = odegenerator([self.bs, self.ns], initsall, return_full=True)
        initsall = initgenerator([1, self.ns]) #[bs, ns, dim]
        neighu, neighv, _ = odegenerator([1, self.ns], initsall, return_full=True)

        weight =  1

        losses = [None for i in range(snapnum)]

        for i in range(snapnum):
            force = self.get_force(fakeu[i][:,None,:], fakev[i][:,None,:],
                                   neighu[i], neighv[i], weight)
            acc = self.get_acc(fakeu[i], fakev[i], faket[i])
            losses[i] = tf.reduce_mean((force - acc)**2)
        return losses


    def get_phi(self, xi, xj):
        '''
        input: (..., dim)
        return: (..., dim)
        '''
        r = tf.norm(xi - xj, axis = -1, keepdims = True)
        rclip = tf.clip_by_value(r, self.clip, np.Inf)
        phi = self.c/(rclip ** (self.dim + self.alpha))
        return phi

    def get_force(self, ix, iv, nx, nv, weight):
        '''
        input: ix: [bs, 1, dim]
            iv: [bs, 1, dim]
            nx: [bs, ns, dim]
            nv: [bs, ns, dim]
            weight: [bs, ns, 1]
        return: (bs, dim)
        '''
        force = self.get_phi(ix, nx) * (nv - iv) # [bs, ns, dim]
        force = tf.reduce_mean(force * weight, axis = 1)
        return force
    
    def get_acc(self, u, v, t):
        '''
        v [bs, dim]
        u [bs, dim]
        t [bs, 1]
        v should be a function of u and t
        return material acceleration = pv/pt + (v dot grad) v
        '''
        vt = tf.concat([tf.gradients(v[:, d], t)[0] for d in range(self.dim)], axis = 1)
        vx = tf.concat([
                tf.reduce_sum(v * tf.gradients(v[:, d], u)[0], axis= 1, keepdims=True) 
                for d in range(self.dim)], axis = 1)
        acc = vt + vx 
        return acc


class MeasureMetric(object):
    def __init__(self, dim, dtype, phase, uref, vref, snapids, metric, umap, vmap, bs, snap_bs):
        self.phase = phase
        self.dim = dim
        self.dtype = dtype
        self.uref = uref
        self.vref = vref
        self.snapids = snapids
        self.snapnum = len(snapids)
        self.metric = metric
        self.umap = umap
        self.vmap = vmap
        self.bs = bs
        self.snap_bs = snap_bs

    def premap(self, raw, ref, mode = 'nml'):
        assert len(raw) == len(ref)
        if mode == 'none':
            return raw
        elif mode == 'nml':
            return [(raw[i] - np.mean(ref[i], axis=0))/np.std(ref[i], axis = 0) for i in range(len(ref))]
        else:
            raise NotImplementedError


    def get_loss(self, initgenerator, odegenerator, trueu, truev, snaprnd):
        if self.metric == "SW":
            self.losses = self.get_loss_SW(initgenerator, odegenerator, trueu, truev)
        elif self.metric == 'WGAN-GP':
            self.losses = self.get_loss_WGANGP(initgenerator, odegenerator, trueu, truev)
        else:
            raise NotImplementedError

        self.snaprnd = snaprnd
        if self.snap_bs > 0:
            self.loss_all = tf.reduce_mean(tf.gather(self.losses, self.snaprnd))
        else:
            self.loss_all = tf.reduce_mean(self.losses)
        return self.loss_all


    def get_loss_SW(self, initgenerator, odegenerator, trueu, truev):

        inits = initgenerator(self.bs)
        fakeu, fakev, _ = odegenerator(self.bs, inits, return_full=True)
        self.fakeu = fakeu
        self.fakev = fakev
        mfakeu = self.premap([fakeu[i] for i in self.snapids], self.uref, self.umap)
        mfakev = self.premap([fakev[i] for i in self.snapids], self.vref, self.vmap)
        mtrueu = self.premap(trueu, self.uref, self.umap)
        mtruev = self.premap(truev, self.vref, self.vmap)

        if self.phase == 'u':
            self.discriminator = [None for i in range(self.snapnum)]
            losses = [None for i in range(self.snapnum)]
            for i in range(self.snapnum):
                true_data = mtrueu[i]
                fake_data = mfakeu[i]
                self.discriminator[i] = SW(self.dim, self.dtype, true_data, fake_data, self.bs)
                losses[i] = self.discriminator[i].Gloss
        elif  self.phase == 'uv':
            self.discriminator = [None for i in range(self.snapnum)]
            losses = [None for i in range(self.snapnum)]
            for i in range(self.snapnum):
                true_data = tf.concat([mtrueu[i], mtruev[i]], axis = 1)
                fake_data = tf.concat([mfakeu[i], mfakev[i]], axis = 1)
                self.discriminator[i] = SW(self.dim * 2, self.dtype, true_data, fake_data, self.bs)
                losses[i] = self.discriminator[i].Gloss
        elif self.phase == 'uu':
            self.discriminator = [None for i in range(self.snapnum-1)]
            losses = [None for i in range(self.snapnum-1)]
            for i in range(self.snapnum-1):
                true_data = tf.concat([mtrueu[i], mtrueu[i+1]], axis = 1)
                fake_data = tf.concat([mfakeu[i], mfakeu[i+1]], axis = 1)
                self.discriminator[i] = SW(self.dim * 2, self.dtype, true_data, fake_data, self.bs)
                losses[i] = self.discriminator[i].Gloss
        else:
            raise NotImplementedError
        return losses



    def get_loss_WGANGP(self, initgenerator, odegenerator, trueu, truev):

        raise NotImplementedError


    def train(self):
        for discriminator in self.discriminator:
            discriminator.train()
