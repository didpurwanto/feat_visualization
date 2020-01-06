import sys
sys.path.append('../../../')
import tensorflow as tf
import numpy as np
import PIL.Image as Image
import random, time
import os, cv2, glob, sys
import matplotlib.pyplot as plt
from six.moves import xrange
from i3d import InceptionI3d
import saliency
from input_visualization import *

                                                                                    
flags = tf.app.flags
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 64, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')                                                                         
flags.DEFINE_integer('rgb_channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 51, 'The num of class')
FLAGS = flags.FLAGS
print(tf.__version__)


def run_visualize():
    pre_model_save_dir = "../best_models/rgb_30000_trainval_split1/i3d_hmdb_model-29999"
    file_list = "../../../list/hmdb_list/testlist1.list"
    pathsave = "../visualization/results/"

    if not os.path.exists(pre_model_save_dir+".index"):
        print('WARNING !!!!! check the path model')
        exit()
    else:   
        print("load model succeed")

        bibnumbers = []
        with open(file_list, "r") as f:
            for line in f:
                bibnumbers.append(line.strip())

        graph = tf.Graph()
        with graph.as_default():
            images_placeholder = tf.placeholder(
                tf.float32, 
                [FLAGS.batch_size, 
                FLAGS.num_frame_per_clib, 
                FLAGS.crop_size, 
                FLAGS.crop_size, 
                FLAGS.rgb_channels]
                )
            with tf.variable_scope('RGB'):
                logits, _ = InceptionI3d(
                               num_classes=FLAGS.classics,
                               spatial_squeeze=True,
                               final_endpoint='Logits', 
                               name='inception_i3d'
                               )(images_placeholder, is_training=False)

            # Create a saver for writing training checkpoints
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()

            # Create a session for running Ops on the Graph
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            sess.run(init)

            # Restore trained model
            saver.restore(sess, pre_model_save_dir)

            neuron_selector = tf.placeholder(tf.int32)
            y = logits[0][neuron_selector]
            prediction = tf.argmax(logits, 1)

        for vid in bibnumbers:
            vid = vid.split(" ")
            vid = vid[0]
            data = vid.split("/")
            pathsave = "./visualization/results/" + data[6] 
            tmp = []
            if not os.path.exists(pathsave):
                os.makedirs(pathsave)

            pathvid = vid+'/i/'
            print(pathvid)

            input_data, num_frames = get_data_vis(
                pathvid,
                num_frames_per_clip=FLAGS.num_frame_per_clib, 
                )

            temp = data_process(
                input_data, 
                FLAGS.crop_size
                )

            tmp.append(temp)
            video_tensor = np.array(tmp).astype(np.float32)

            out_feature = sess.run(
                logits,
                feed_dict={images_placeholder: video_tensor}
                )

            prediction_class = sess.run(
                prediction,
                feed_dict={images_placeholder: video_tensor}
                )[0]

            guided_backprop = saliency.GuidedBackprop(
                graph, sess, y, images_placeholder
                )

            # Compute the vanilla mask and the smoothed mask.
            vanilla_guided_backprop_mask_3d = guided_backprop.GetMask(
                video_tensor[0], 
                feed_dict = {neuron_selector: prediction_class}
                )
            smoothgrad_guided_backprop_mask_3d = guided_backprop.GetSmoothedMask(
                video_tensor[0], 
                feed_dict = {neuron_selector: prediction_class}
                )

            vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(
                vanilla_guided_backprop_mask_3d
                )
            smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(
                smoothgrad_guided_backprop_mask_3d
                )

            # xrai_object = saliency.XRAI(
            #     graph, sess, y, images_placeholder
            #     )
            # xrai_attributions = xrai_object.GetMask(
            #     video_tensor[0], 
            #     feed_dict={neuron_selector: prediction_class}
            #     )


            
            resize_ori = data_resize(
                input_data, 
                FLAGS.crop_size
                )

            # save results
            vis_ori(pathsave, resize_ori, FLAGS.num_frame_per_clib)
            vis_maps(pathsave, smoothgrad_mask_grayscale, FLAGS.num_frame_per_clib)
            # vis_heat(pathsave, xrai_attributions, FLAGS.num_frame_per_clib)
            

def main(_):
    run_visualize()


if __name__ == '__main__':
    tf.app.run()

  

