import tensorflow as tf
import os
import numpy as np
import cv2 as cv
import json

from models import H_estimator
from utils import DataLoader, load, save
import constant
# skimage.measure.compare_psnr/compare_ssim were removed in newer skimage; use
# the skimage.metrics API (available in scikit-image 0.16.2 used by this env).
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

slim = tf.contrib.slim

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
test_folder = constant.TEST_FOLDER
snapshot_dir = constant.SNAPSHOT_DIR + '/model.ckpt-' + str(constant.HOMO_CKPT_STEP)
batch_size = constant.TEST_BATCH_SIZE
metrics_out = constant.METRICS_OUT

# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    
    test_inputs = tf.placeholder(shape=[batch_size, 128, 128, 3 * 2], dtype=tf.float32)
    print('test inputs = {}'.format(test_inputs))



with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_net1_f, test_net2_f, test_net3_f, test_warp2_H1, test_warp2_H2, test_warp2_H3, test_one_warp_H1, test_one_warp_H2, test_one_warp_H3 = H_estimator(test_inputs, test_inputs, False)
   


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
        # number of test pairs is derived from the dataset instead of being hardcoded
        length = data_loader.datas['input1']['length']
        if constant.LIMIT > 0:
            length = min(length, constant.LIMIT)
        input1_frames = data_loader.datas['input1']['frame']
        psnr_list = []
        ssim_list = []
        per_pair = []

        for i in range(0, length):
            #load test data
            input_clip = np.expand_dims(data_loader.get_data_clips(i, 128, 128), axis=0)
            
            # inference
            _, _, _, _, _, warp, _, _, warp_one = sess.run([test_net1_f, test_net2_f, test_net3_f, test_warp2_H1, test_warp2_H2, test_warp2_H3, test_one_warp_H1, test_one_warp_H2, test_one_warp_H3], feed_dict={test_inputs: input_clip})
            
            
            warp = (warp+1) * 127.5    
            warp = warp[0] 
            warp_one = warp_one[0]
            input1 = (input_clip[...,0:3]+1) * 127.5    
            input1 = input1[0]
            input2 = (input_clip[...,3:6]+1) * 127.5    
            input2 = input2[0]
            
            # compute psnr/ssim on the overlapping (aligned) region only
            psnr = compare_psnr(input1*warp_one, warp*warp_one, data_range=255)
            ssim = compare_ssim(input1*warp_one, warp*warp_one, data_range=255, multichannel=True)

            
            print('i = {} / {}, psnr = {:.6f}'.format( i+1, length, psnr))
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            per_pair.append({
                'index': i + 1,
                'name': os.path.basename(input1_frames[i]),
                'psnr': float(psnr),
                'ssim': float(ssim),
            })
            
            
        print("===================Results Analysis==================")
        # split into top 30% / 30-60% / 60-100% dynamically so this works for
        # any test-set size (UDIS-D = 1106, StitchBench/General ~ 100+).
        s30 = int(round(length * 0.3))
        s60 = int(round(length * 0.6))

        psnr_sorted = sorted(psnr_list, reverse=True)
        psnr_30 = float(np.mean(psnr_sorted[0:s30])) if s30 > 0 else 0.0
        psnr_60 = float(np.mean(psnr_sorted[s30:s60])) if s60 > s30 else 0.0
        psnr_100 = float(np.mean(psnr_sorted[s60:])) if length > s60 else 0.0
        print("top 30%", psnr_30)
        print("top 30~60%", psnr_60)
        print("top 60~100%", psnr_100)
        print('average psnr:', float(np.mean(psnr_list)))

        ssim_sorted = sorted(ssim_list, reverse=True)
        ssim_30 = float(np.mean(ssim_sorted[0:s30])) if s30 > 0 else 0.0
        ssim_60 = float(np.mean(ssim_sorted[s30:s60])) if s60 > s30 else 0.0
        ssim_100 = float(np.mean(ssim_sorted[s60:])) if length > s60 else 0.0
        print("top 30%", ssim_30)
        print("top 30~60%", ssim_60)
        print("top 60~100%", ssim_100)
        print('average ssim:', float(np.mean(ssim_list)))

        if metrics_out:
            summary = {
                'count': length,
                'psnr_average': float(np.mean(psnr_list)),
                'psnr_top_30': psnr_30,
                'psnr_top_30_60': psnr_60,
                'psnr_top_60_100': psnr_100,
                'ssim_average': float(np.mean(ssim_list)),
                'ssim_top_30': ssim_30,
                'ssim_top_30_60': ssim_60,
                'ssim_top_60_100': ssim_100,
            }
            out_dir = os.path.dirname(metrics_out)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(metrics_out, 'w') as f:
                json.dump({'summary': summary, 'per_pair': per_pair}, f, indent=2)
            print('Wrote metrics to {}'.format(metrics_out))

    inference_func(snapshot_dir)
