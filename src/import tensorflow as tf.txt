import tensorflow as tf
import numpy as np
import tfutil
import metric
import tfutil as tfu

import numpy as np
import tensorflow as tf


class RCAN:
@@ -34,7 +35,7 @@ def __init__(self,
                 beta2=.999,                               # Adam beta2 value
                 opt_eps=1e-8,                             # Adam epsilon value
                 eps=1.1e-5,                               # epsilon
                 tf_log="./model/",                        # path saved tensor summary/model
                 tf_log="./model/",                        # path saved tensor summary / model
                 n_gpu=1,                                  # number of GPU
                 ):
        self.sess = sess
@@ -110,16 +111,16 @@ def setup(self):
        elif self.activation == 'elu':
            self.act = tf.nn.elu
        else:
            raise NotImplementedError("[-] Not supported activation function (%s)" % self.activation)
            raise NotImplementedError("[-] Not supported activation function {}".format(self.activation))

        # Optimizer
        if self.optimizer == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr,
                                              beta1=self.beta1, beta2=self.beta2, epsilon=self.opt_eps)
        elif self.optimizer == 'sgd':  # sgd + m with nesterov
        elif self.optimizer == 'sgd':  # sgd + m with nestrov
            self.opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.momentum, use_nesterov=True)
        else:
            raise NotImplementedError("[-] Not supported optimizer (%s)" % self.optimizer)
            raise NotImplementedError("[-] Not supported optimizer {}".format(self.optimizer))

    def image_processing(self, x, sign, name):
        with tf.variable_scope(name):
@@ -143,24 +144,24 @@ def channel_attention(self, x, f, reduction, name):
        with tf.variable_scope("CA-%s" % name):
            skip_conn = tf.identity(x, name='identity')

            x = tfutil.adaptive_global_average_pool_2d(x)
            x = tfu.adaptive_global_average_pool_2d(x)

            x = tfutil.conv2d(x, f=f // reduction, k=1, name="conv2d-1")
            x = tfu.conv2d(x, f=f // reduction, k=1, name="conv2d-1")
            x = self.act(x)

            x = tfutil.conv2d(x, f=f, k=1, name="conv2d-2")
            x = tfu.conv2d(x, f=f, k=1, name="conv2d-2")
            x = tf.nn.sigmoid(x)
            return tf.multiply(skip_conn, x)

    def residual_channel_attention_block(self, x, f, kernel_size, reduction, use_bn, name):
        with tf.variable_scope("RCAB-%s" % name):
            skip_conn = tf.identity(x, name='identity')

            x = tfutil.conv2d(x, f=f, k=kernel_size, name="conv2d-1")
            x = tfu.conv2d(x, f=f, k=kernel_size, name="conv2d-1")
            x = tf.layers.BatchNormalization(epsilon=self._eps, name="bn-1")(x) if use_bn else x
            x = self.act(x)

            x = tfutil.conv2d(x, f=f, k=kernel_size, name="conv2d-2")
            x = tfu.conv2d(x, f=f, k=kernel_size, name="conv2d-2")
            x = tf.layers.BatchNormalization(epsilon=self._eps, name="bn-2")(x) if use_bn else x

            x = self.channel_attention(x, f, reduction, name="RCAB-%s" % name)
@@ -173,7 +174,7 @@ def residual_group(self, x, f, kernel_size, reduction, use_bn, name):
            for i in range(self.n_res_blocks):
                x = self.residual_channel_attention_block(x, f, kernel_size, reduction, use_bn, name=str(i))

            x = tfutil.conv2d(x, f=f, k=kernel_size, name='rg-conv-1')
            x = tfu.conv2d(x, f=f, k=kernel_size, name='rg-conv-1')
            return x + skip_conn  # tf.math.add(x, skip_conn)

    def up_scaling(self, x, f, scale_factor, name):
@@ -186,13 +187,13 @@ def up_scaling(self, x, f, scale_factor, name):
        """
        with tf.variable_scope(name):
            if scale_factor == 3:
                x = tfutil.conv2d(x, f * 9, k=1, name='conv2d-image_scaling-0')
                x = tfutil.pixel_shuffle(x, 3)
                x = tfu.conv2d(x, f * 9, k=1, name='conv2d-image_scaling-0')
                x = tfu.pixel_shuffle(x, 3)
            elif scale_factor & (scale_factor - 1) == 0:  # is it 2^n?
                log_scale_factor = int(np.log2(scale_factor))
                for i in range(log_scale_factor):
                    x = tfutil.conv2d(x, f * 4, k=1, name='conv2d-image_scaling-%d' % i)
                    x = tfutil.pixel_shuffle(x, 2)
                    x = tfu.conv2d(x, f * 4, k=1, name='conv2d-image_scaling-%d' % i)
                    x = tfu.pixel_shuffle(x, 2)
            else:
                raise NotImplementedError("[-] Not supported scaling factor (%d)" % scale_factor)
            return x
@@ -202,19 +203,19 @@ def residual_channel_attention_network(self, x, f, kernel_size, reduction, use_b
            x = self.image_processing(x, sign=-1, name='pre-processing')

            # 1. head
            head = tfutil.conv2d(x, f=f, k=kernel_size, name="conv2d-head")
            head = tfu.conv2d(x, f=f, k=kernel_size, name="conv2d-head")

            # 2. body
            x = head
            for i in range(self.n_res_groups):
                x = self.residual_group(x, f, kernel_size, reduction, use_bn, name=str(i))

            body = tfutil.conv2d(x, f=f, k=kernel_size, name="conv2d-body")
            body = tfu.conv2d(x, f=f, k=kernel_size, name="conv2d-body")
            body += head  # tf.math.add(body, head)

            # 3. tail
            x = self.up_scaling(body, f, scale, name='up-scaling')
            tail = tfutil.conv2d(x, f=self.n_channel, k=kernel_size, name="conv2d-tail")  # (-1, 384, 384, 3)
            tail = tfu.conv2d(x, f=self.n_channel, k=kernel_size, name="conv2d-tail")  # (-1, 384, 384, 3)

            x = self.image_processing(tail, sign=1, name='post-processing')
            return x
@@ -236,8 +237,8 @@ def build_model(self):
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

        # metrics
        self.psnr = tf.reduce_mean(metric.psnr(self.output, self.x_hr, m_val=1.))
        self.ssim = tf.reduce_mean(metric.ssim(self.output, self.x_hr, m_val=1.))
        self.psnr = tf.reduce_mean(metric.psnr(self.output, self.x_hr, m_val=1))
        self.ssim = tf.reduce_mean(metric.ssim(self.output, self.x_hr, m_val=1))

        # summaries
        tf.summary.image('lr', self.x_lr, max_outputs=self.batch_size)
        tf.summary.image('hr', self.x_hr, max_outputs=self.batch_size)
        tf.summary.image('generated-hr', self.output, max_outputs=self.batch_size)
        tf.summary.scalar("loss/l1_loss", self.loss)
        tf.summary.scalar("metric/psnr", self.psnr)
        tf.summary.scalar("metric/ssim", self.ssim)
        tf.summary.scalar("misc/lr", self.lr)
        # merge summary
        self.merged = tf.summary.merge_all()
        # model saver
        self.saver = tf.train.Saver(max_to_keep=1)
        self.best_saver = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.tf_log, self.sess.graph)