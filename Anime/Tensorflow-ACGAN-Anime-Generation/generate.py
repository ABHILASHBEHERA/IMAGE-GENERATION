import pickle 
import os 
from libs.util import *
import tensorflow as tf
import numpy as np
import sys
import os
import scipy.stats as stats
from scipy import misc
import random
from  network.model_srresnet import Generator_srresnet
import argparse
from google.colab.patches import cv2_imshow
# %matplotlib inline
from matplotlib import pyplot as plt
from PIL import Image as im

tf.flags.DEFINE_integer("z_dim", 100, "noise dimension")
tf.flags.DEFINE_string("img_dir", "/content/drive/MyDrive/Computational Creativity/Anime/output_imgs", "test image directory")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
FLAGS(sys.argv)
test_size = 5
hair_map = {}
eye_map = {}

for idx, h in enumerate(hair_color):
    hair_map[h] = idx

for idx, e in enumerate(eye_color):
    eye_map[e] = idx

TEST_PATH = './testing_text.txt'
MODEL_PATH = '/content/drive/MyDrive/Computational Creativity/Anime/models/model-350500.data-00000-of-00001'

if __name__ == '__main__':

    ap = argparse.ArgumentParser(description= 'Generate anime image')

    ap.add_argument('-t', '--test_file', type=str, help='Path to the test file')
    ap.add_argument('-m', '--model', type=str, help='.ckpt file of the model. If -t option is used, evaluate this model. Otherwise, train it.')

    ap.set_defaults(test_file = TEST_PATH, model = MODEL_PATH)

    args = ap.parse_args()

    # TEST_PATH = args.test_file
    TEST_PATH = '/content/drive/MyDrive/Computational Creativity/Anime/datasets/sample_testing_text.txt'
    MODEL_PATH = args.model

    seq = tf.placeholder(tf.float32, [None, len(hair_color)+len(eye_color)], name="seq")      
    z = tf.placeholder(tf.float32, [None, FLAGS.z_dim])

    # g_net = Generator_srresnet(  embedding_size=100, 
    #                     hidden_size=100,
    #                     img_row=96,
    #                     img_col=96, train = False)
    
    g_net = Generator_srresnet(hidden_size=100,
                        img_row=96,
                        img_col=96, train = False)
    result = g_net(seq, z)

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=None)

    # saver.restore(sess, save_path=MODEL_PATH)
    saver = tf.train.import_meta_graph('/content/drive/MyDrive/Computational Creativity/Anime/models/model-350500.meta')
    saver.restore(sess, "/content/drive/MyDrive/Computational Creativity/Anime/models/model-350500")

    z_sampler = stats.truncnorm((-1 - 0.) / 1., (1 - 0.) / 1., loc=0., scale=1)

    test = load_test(TEST_PATH, hair_map, eye_map)

    z_noise = z_sampler.rvs([test_size, 100])
    

    for idx, t in enumerate(test):
        
        t = np.expand_dims(t, axis=0)
        cond = np.repeat(t, test_size, axis=0)
        feed_dict = {seq: cond,  z:z_noise}

        sampler = tf.identity(g_net(seq, z, reuse=True, train=False), name='sampler')

        f_imgs = sess.run(sampler, feed_dict=feed_dict)
        
        # data = im.fromarray(f_imgs[0],'RGB')
        # data.save('/content/drive/MyDrive/Computational Creativity/Anime/test_pic.png')
        # data.show()
        # print(f_imgs.shape)
        # for i in range(f_imgs.shape[0]):
        #     plt.imshow(f_imgs[i], interpolation='nearest')
        #     plt.show()
        dump_img(FLAGS.img_dir, f_imgs, idx+1)
