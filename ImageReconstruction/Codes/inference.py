import tensorflow as tf
import numpy as np
import cv2
import os

from models import reconstruction
from utils import DataLoader, load, save
import constant



os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
test_folder = constant.TEST_FOLDER
snapshot_dir = constant.SNAPSHOT_DIR + '/model.ckpt-' + str(constant.RECON_CKPT_STEP)
batch_size = constant.TEST_BATCH_SIZE
result_out = constant.RESULT_OUT



# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    test_inputs = tf.placeholder(shape=[batch_size, None, None, 3 * 2], dtype=tf.float32)
    print('test inputs = {}'.format(test_inputs))


# define testing generator function
with tf.variable_scope('Reconstruction', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    lr_test_stitched, hr_test_stitched = reconstruction(test_inputs)
 


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)

    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        # number of pairs derived from the dataset instead of being hardcoded
        length = data_loader.images['warp1']['length']
        if constant.LIMIT > 0:
            length = min(length, constant.LIMIT)
        if not os.path.exists(result_out):
            os.makedirs(result_out)

        for i in range(0, length):
            input_clip = np.expand_dims(data_loader.get_image_clips(i), axis=0)
            _, stitch_result = sess.run([lr_test_stitched, hr_test_stitched], feed_dict={test_inputs: input_clip})
            
            stitch_result = (stitch_result+1) * 127.5    
            stitch_result = stitch_result[0]
            path = os.path.join(result_out, str(i+1).zfill(6) + ".jpg")
            cv2.imwrite(path, stitch_result)
            print('i = {} / {}'.format( i+1, length))
            
        print("===================DONE!==================")  

    inference_func(snapshot_dir)
