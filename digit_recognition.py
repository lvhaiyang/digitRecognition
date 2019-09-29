#!encoding=utf8

import tensorflow as tf
from PIL import Image, ImageFilter


class DigitRecognition(object):

    def __init__(self):
        # self.saver = tf.train.Saver()  # 定义saver
        self.model_path = './model/model.ckpt'
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y_ = tf.placeholder(tf.float32, [None, 10])

        self.W_conv1 = self.weight_variable([5, 5, 1, 32])
        self.b_conv1 = self.bias_variable([32])

        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        self.W_conv2 = self.weight_variable([5, 5, 32, 64])
        self.b_conv2 = self.bias_variable([64])

        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)

        self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = self.bias_variable([1024])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.keep_prob = tf.placeholder("float")
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.W_fc2 = self.weight_variable([1024, 10])
        self.b_fc2 = self.bias_variable([10])

        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

        # 训练用的参数
        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))

        # 识别用的参数
        self.prediction = tf.argmax(self.y_conv, 1)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def imageprepare(self, image_path):
        im = Image.open(image_path)  # 读取的图片所在路径，注意是28*28像素
        # plt.imshow(im)  # 显示需要识别的图片
        # plt.show()
        im = im.convert('L')
        tv = list(im.getdata())
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]
        return tva

    def train(self):
        saver = tf.train.Saver()
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)  # MNIST数据集所在路径
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(20000):
                batch = mnist.train.next_batch(50)
                if i % 100 == 0:
                    train_accuracy = self.accuracy.eval(feed_dict={
                        self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
            saver.save(sess, self.model_path)  # 模型储存位置

            print('test accuracy %g' % self.accuracy.eval(feed_dict={
                self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0}))

    def run(self, image_path):
        saver = tf.train.Saver()
        result = self.imageprepare(image_path)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.model_path)  # 使用模型，参数和之前的代码保持一致
            predint = self.prediction.eval(feed_dict={self.x: [result], self.keep_prob: 1.0}, session=sess)

            print('识别结果: {0}'.format(predint[0]))


if __name__ == '__main__':
    dr = DigitRecognition()
    # dr.train()
    dr.run('./test_image/3.png')
