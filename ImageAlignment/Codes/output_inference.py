import tensorflow as tf
import os
import numpy as np
import cv2


from models import H_estimator, output_H_estimator
from utils import DataLoader, load, save
import constant
import skimage


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
test_folder = constant.TEST_FOLDER
snapshot_dir = constant.SNAPSHOT_DIR + '/model.ckpt-' + str(constant.HOMO_CKPT_STEP)
batch_size = constant.TEST_BATCH_SIZE
# output directory for the coarsely aligned images + content masks
warp_out = constant.WARP_OUT

# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    test_inputs = tf.placeholder(shape=[batch_size, None, None, 3 * 2], dtype=tf.float32)
    test_size = tf.placeholder(shape=[batch_size, 2, 1], dtype=tf.float32)
    print('test inputs = {}'.format(test_inputs))
    print('test size = {}'.format(test_size))



with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_coarsealignment = output_H_estimator(test_inputs, test_size, False)
    


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.Session(config=config) as sess:


    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)

    def make_dirs(base):
        for sub in ['warp1', 'warp2', 'mask1', 'mask2']:
            d = os.path.join(base, sub)
            if not os.path.exists(d):
                os.makedirs(d)

    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")

        print("------------------------------------------")
        print("generating aligned images + masks")
        print("input  : {}".format(test_folder))
        print("output : {}".format(warp_out))
        make_dirs(warp_out)

        data_loader = DataLoader(test_folder)
        length = data_loader.datas['input1']['length']
        if constant.LIMIT > 0:
            length = min(length, constant.LIMIT)
        for i in range(0, length):
            input_clip = np.expand_dims(data_loader.get_data_clips(i, None, None), axis=0)
            size_clip = np.expand_dims(data_loader.get_size_clips(i), axis=0)

            coarsealignment = sess.run(test_coarsealignment, feed_dict={test_inputs: input_clip, test_size: size_clip})

            coarsealignment = coarsealignment[0]
            warp1 = (coarsealignment[..., 0:3] + 1.) * 127.5
            warp2 = (coarsealignment[..., 3:6] + 1.) * 127.5
            mask1 = coarsealignment[..., 6:9] * 255
            mask2 = coarsealignment[..., 9:12] * 255

            name = str(i + 1).zfill(6) + ".jpg"
            cv2.imwrite(os.path.join(warp_out, 'warp1', name), warp1)
            cv2.imwrite(os.path.join(warp_out, 'warp2', name), warp2)
            cv2.imwrite(os.path.join(warp_out, 'mask1', name), mask1)
            cv2.imwrite(os.path.join(warp_out, 'mask2', name), mask2)

            print('i = {} / {}'.format(i + 1, length))

        print("-----------done--------------")
        print("------------------------------------------")

    inference_func(snapshot_dir)
