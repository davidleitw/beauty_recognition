import os
import cv2
import numpy as np
import tensorflow as tf
from six import string_types, iteritems

def layer(operator):
    def layer_decorated(self, *args, **kwargs):
        name = kwargs.setdefault('name', self.get_unique_name(operator.__name__))
        print('layer name = {}'.format(name))

        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer {}'.format(name))
        
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]

        else:
            layer_input = list(self.terminals)

        layer_output = operator(self, layer_input, *args, **kwargs)
        self.layers[name] = layer_output
        self.feed(layer_output)
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.input = inputs 
        self.terminals = []
        self.layers = dict(self.input)
        self.train = trainable
        self.setup()

    def layer(self, operator):
        def layer_decorated(self, *args, **kwargs):
            name = kwargs.setdefault('name', self.get_unique_name(operator.__name__)) # __name__ is string type
            if len(self.terminals) == 0:
                raise RuntimeError('No input variables found for layer {}'.format(name))
            elif len(self.terminals) == 1:
                layer_input = self.terminals[0]
            else:
                layer_input = list(self.terminals)

            layer_output = operator(self, layer_input, *args, **kwargs)
            self.layers[name] = layer_output
            self.feed(layer_output)
            return self
        return layer_decorated

    def setup(self):
        # 類似『虛擬函式』, 繼承之後一定要實現, 不然會拋出error
        raise NotImplementedError('Must be implemented by the subclass.')

    def load_data(self, data_path, session, ignore_missing=True):
        # load network weights
        # encoding = 'latin1' 預防編碼問題
        data_dict = np.load(data_path, encoding='latin1').item() # .item()用在純量上面
        print('loading data, data_dict = {}'.format(data_dict))
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        value = tf.get_variable(param_name)
                        session.run(value.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise ValueError
                        
    def feed(self, *args):
        assert len(args) != 0
        # 使用feed()函式將一層一層的指令input進去建構我們的nn
        self.terminals = []
        print('feed function, args = {}'.format(args))
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed {}'.format(fed_layer))
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        return self.terminals[-1]
    
    def get_unique_name(self, prefix):
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        print('In self.get_unique_name, prefix = {}, ident = {}'.format(prefix, ident))
        return '{}__{}'.format(prefix, ident)
    
    def make_var(self, name, shape):
        return tf.get_variable(name, shape, trainable=self.train)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, inp, k_h, k_w, c_o, s_h, s_w, conv_name, relu=True, padding='SAME',
             group=1, biased=True):
        self.validate_padding(padding)
        c_i = int(inp.get_shape()[-1])
        assert c_i % group == 0
        assert c_o % group == 0

        # Convolution for a given input and kernal
        convolve = lambda i, k : tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(conv_name) as scope:
            kernel = self.make_var(name='weights', shape=[k_h, k_w, (c_i//group), c_o])
            output = convolve(inp, kernel)

            if biased is True:
                biased = self.make_var(name='biased', shape=[c_o])
                output = tf.nn.biasadd(output, biased)
            if relu is True:
                output = self.nn.relu(output, name=scope.name)

            return output
    
    @layer
    def prelu(self, inp, name):
        with tf.variable_scope(name) as scope:
            i = int(inp.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i, ))           
            output = tf.nn.relu(inp) + tf.multiply(alpha, -(tf.nn.relu(-inp)))
        return output
    
    @layer 
    def max_pooling(self, inp, k_h, k_w, s_h, s_w, name, padding='SAME'):
        self.validate_padding(padding)
        return tf.nn.max_pool(inp, ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1], padding=padding,
                              name=name)

    @layer 
    def fc(self, inp, num_out, name, relu=True):
        with tf.variable_scope(name):
            input_shape = inp.get_shape()




















                    






                    







                        










