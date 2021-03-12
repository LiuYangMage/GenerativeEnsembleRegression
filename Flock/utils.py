import numpy as np
import tensorflow as tf

class Net(object):
    def __init__(self, layer_dims, name, dtype, act = tf.nn.tanh):
        self.L = len(layer_dims)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.W = [tf.get_variable('W_{}'.format(l), [layer_dims[l-1], layer_dims[l]], 
                dtype=dtype, initializer=tf.contrib.layers.xavier_initializer()) for l in range(1, self.L)]
            self.b = [tf.get_variable('b_{}'.format(l), [1,layer_dims[l]], 
                dtype=dtype, initializer=tf.zeros_initializer()) for l in range(1, self.L)]
        self.act = act

    def __call__(self, x):
        A = x
        for i in range(self.L-2):
            A = self.act(tf.matmul(A, self.W[i]) + self.b[i])
        return tf.matmul(A, self.W[-1]) + self.b[-1]

    def trainable_variables(self):
        return self.W + self.b

class SW(object):
    def __init__(self, dim, dtype, true_data, fake_data, bs, num_projections = 1000):
        self.dim = dim
        self.dtype = dtype
        self.num_projections = 1 if (self.dim == 1) else num_projections
        self.Gloss = self.metric(true_data, fake_data, bs)

    def trainable_variables(self):
        return []

    def train(self):
        pass

    def metric(self, true_data, fake_data, bs):
        theta = tf.nn.l2_normalize(tf.random_normal(shape=[self.dim, self.num_projections], dtype = self.dtype), axis=0)
        projected_true = tf.transpose(tf.matmul(true_data, theta))
        projected_fake = tf.transpose(tf.matmul(fake_data, theta))

        sorted_true, _ = tf.nn.top_k(projected_true,bs)
        sorted_fake, _ = tf.nn.top_k(projected_fake,bs)

        return tf.reduce_mean(tf.square(sorted_true - sorted_fake))


class ODEFlow(object):
    def __init__(self, dim, dtype, grad, steps, dt, name, scheme='first', hidden=None):
        if hidden is None:
            hidden = 3*[128]
        if grad:
            layer_dims = [dim + 1] + hidden + [1]
        else:
            layer_dims = [dim + 1] + hidden + [dim]
        self.net = Net(layer_dims, name, dtype = dtype, act=tf.nn.tanh)

        self.dim = dim
        self.dtype = dtype
        self.grad = grad
        self.steps = steps
        self.dt = dt
        self.T = steps * dt
        self.scheme = scheme

    def __call__(self, bs, z = None, den0 = None, return_den = False, return_full = False):
        '''
        bs is a list. For example, x shape [a, b, dim], then bs = [a, b]
        den0: bs
        '''
        return self.generate(bs, z, den0, return_den, return_full)

    def trainable_variables(self):
        return self.net.trainable_variables()

    def get_v(self, u, t):
        if self.grad:
            return tf.gradients(self.net(tf.concat([u, t], axis = -1)), u)[0]
        else:
            return self.net(tf.concat([u, t], axis = -1))

    def generate(self, bs, z, den0, return_den, return_full):
        if not (isinstance(bs, list)): bs = [bs] 
        nin = False
        if z is None:
            nin = True
            z = tf.random.normal(bs+[self.dim], 0, 1, dtype = self.dtype)
        u = [None for i in range(self.steps + 1)]
        v = [None for i in range(self.steps + 1)]
        t = [i * self.dt * tf.ones(bs+[1], dtype = self.dtype) for i in range(self.steps + 1)]
        if return_den:
            div = [None for i in range(self.steps + 1)]

        for i in range(self.steps+1):
            print('.', end = '', flush=True)
            if i == 0:
                u[i] = z 
            else:
                if self.scheme == 'first':
                    u[i] = u[i-1] + self.dt * v[i-1]
                elif self.scheme == 'midpoint':
                    midu = u[i-1] + 0.5 * self.dt * v[i-1]
                    midv = self.get_v(midu, t[i-1]+0.5*self.dt)
                    u[i] = u[i-1] + self.dt * midv
            v[i] = self.get_v(u[i], t[i])
            if return_den:
                weight = 0.5 * self.dt if (i == 0 or self.steps) else self.dt
                div[i] = weight * tf.reduce_sum([tf.gradients(v[i][...,j], u[i])[0][...,j] for j in range(self.dim)], axis = 0) 
        print('Generated {} steps'.format(self.steps+1))
        if return_den:
            if nin == True:
                den0 = get_gaussian_density(u[-1], 1.0, self.dim)
            else:
                assert not (den0 is None)
            if return_full:
                raise NotImplementedError
            else: 
                den = den0 * tf.exp(-tf.reduce_sum(div, axis = 0))
                return u[-1], den
        else:
            if return_full:
                return u, v, t
            else:
                return u[-1]

    def get_den(self, x, bs, den0fun=None, args=None, thisstep=None, **kwargs):
        if thisstep is None:
            thisstep = self.steps
        if not (isinstance(bs, list)): bs = [bs] 
        u = [None for i in range(thisstep + 1)]
        v = [None for i in range(thisstep + 1)]
        t = [self.T - i * self.dt * tf.ones(bs +[1], dtype = self.dtype) for i in range(thisstep + 1)]
        div = [None for i in range(thisstep + 1)]

        for i in range(thisstep + 1):
            u[i] = x if i == 0 else u[i-1] - self.dt * v[i-1]
            v[i] = self.get_v(u[i], t[i])
            weight = 0.5 * self.dt if (i == 0 or thisstep) else self.dt
            div[i] = weight * tf.reduce_sum([tf.gradients(v[i][...,j], u[i])[0][...,j] for j in range(self.dim)], axis = 0) # shape of bs
        
        if den0fun is None:
            den0 = get_gaussian_density(u[-1], 1.0, self.dim)
        else:
            if args is None:
                args = {}
            den0 = den0fun(u[-1], **args)

        if thisstep == 0:
            return den0
        else:
            den = den0 * tf.exp(-tf.reduce_sum(div, axis = 0))
            return den



def get_gaussian_density(x, sigma, dim):
    '''
    x: [..., dim]
    '''
    return 1/(np.sqrt(2 * np.pi) * sigma)**dim * tf.exp( - 0.5 * tf.reduce_sum(x**2, axis = -1)/sigma**2)