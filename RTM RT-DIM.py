"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import imageio
#from scipy.misc import imread
#from scipy.misc import imsave
import time
import tensorflow as tf
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2


from PIL import Image

from pylab import *

slim = tf.contrib.slim

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')



tf.flags.DEFINE_string(
    'checkpoint_path_inception_v3', './models/inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_v4', './models/inception_v4.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet', './models/resnet_v2_101.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_inception_resnet_v2', './models/inception_resnet_v2_2016_08_30.ckpt', 'Path to checkpoint for inception network.')


tf.flags.DEFINE_string(
    'checkpoint_path_adv_inception_v3', './models/adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens3_adv_inception_v3', './models/ens3_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens4_adv_inception_v3', './models/ens4_adv_inception_v3_rename.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'checkpoint_path_ens_adv_inception_resnet_v2', './models/ens_adv_inception_resnet_v2_rename.ckpt', 'Path to checkpoint for inception network.')


tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')
tf.flags.DEFINE_integer(
    'image_resize', 330, 'Height of each input images.')
tf.flags.DEFINE_float(
    'prob', 0.5, 'probability of using diverse inputs.')

tf.flags.DEFINE_integer(
    'batch_size', 15, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'momentum', 1, 'Momentum.')


tf.flags.DEFINE_string(
    'GPU_ID', '0,1', 'which GPU to use.')


FLAGS = tf.flags.FLAGS

# =============================================================================
# se = int(time.time())
# print (se)
# np.random.seed(se)
# tf.set_random_seed(se)
# =============================================================================
np.random.seed(1607906060)
tf.set_random_seed(1607906060)

print("print all settings\n")
print(FLAGS.master)
#print(FLAGS.__dict__)
print ("hello world")
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.GPU_ID


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  
  
#  for filepath in sorted(tf.io.gfile.glob(os.path.join(input_dir, '*.png')))[:20]:
#    with tf.io.gfile.GFile(filepath, "rb") as f:
#       images[idx, :, :, :] = imageio.imread(f, pilmode='RGB').astype(np.float)*2.0/255.0 - 1.0
  
  
  
  for filepath in sorted(tf.io.gfile.glob(os.path.join(input_dir, '*.png')))[:1000]:
    with tf.io.gfile.GFile(filepath, "rb") as f:
      image = imageio.imread(f, pilmode='RGB').astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.io.gfile.GFile(os.path.join(output_dir, filename), 'w') as f:
      imageio.imsave(f, Image.fromarray(uint8((images[i, :, :, :] + 1.0) * 0.5 * 255)), format='png')
      
      
      
beta1 = 0.99
beta2 = 0.999
num_iter1 = FLAGS.num_iter
weight=0
t = np.arange(1,num_iter1+0.1,1)
y1 = np.sqrt(1 - beta2**t) / (1 - beta1**t)

for x1 in y1:
    weight+=x1
      


def graph(x, y, i, x_max, x_min, grad, grad2):
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  num_iter = FLAGS.num_iter
  batch_size = FLAGS.batch_size
  alpha = eps / num_iter
  alpha_norm2 = eps * np.sqrt(299 * 299 * 3) / num_iter
  momentum = FLAGS.momentum

  delta = 1e-08
  num_classes = 1001


  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_v3, end_points_v3 = inception_v3.inception_v3(
        input_diversity(x), num_classes=num_classes, is_training=False)

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
        input_diversity(x), num_classes=num_classes, is_training=False, scope='AdvInceptionV3')

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
        input_diversity(x), num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')

  with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
        input_diversity(x), num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')

  with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
    logits_v4, end_points_v4 = inception_v4.inception_v4(
        input_diversity(x), num_classes=num_classes, is_training=False)

  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_res_v2, end_points_res_v2 = inception_resnet_v2.inception_resnet_v2(
        input_diversity(x), num_classes=num_classes, is_training=False)

  with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
    logits_ensadv_res_v2, end_points_ensadv_res_v2 = inception_resnet_v2.inception_resnet_v2(
        input_diversity(x), num_classes=num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')

  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits_resnet, end_points_resnet = resnet_v2.resnet_v2_101(
        input_diversity(x), num_classes=num_classes, is_training=False)
     
  
  pred = tf.argmax(end_points_v3['Predictions'], 1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)  

  logits = logits_v3
  cross_entropy = tf.compat.v1.losses.softmax_cross_entropy(one_hot,logits)
        
  auxlogits = end_points_v3['AuxLogits']
      
  cross_entropy += tf.compat.v1.losses.softmax_cross_entropy(one_hot,auxlogits,label_smoothing=0.0,weights=0.4)

  noise = tf.gradients(cross_entropy, x)[0]
  
  noise2 = grad2

  #===============================MI-FGSM=======================================

  noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
  noise = momentum * grad + noise
  x = x + alpha * tf.sign(noise)
   
  
  #===============================I-FGSM=======================================

# =============================================================================
#   x = x + alpha * tf.sign(noise)
# =============================================================================
  
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, y, i, x_max, x_min, noise, noise2


def stop(x, y, i, x_max, x_min, grad, grad2):
  num_iter = FLAGS.num_iter
  return tf.less(i, num_iter)


def input_diversity(input_tensor):
# =============================================================================
#   rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
#   rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#   h_rem = FLAGS.image_resize - rnd
#   w_rem = FLAGS.image_resize - rnd
#   pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
#   pad_bottom = h_rem - pad_top
#   pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
#   pad_right = w_rem - pad_left
#   padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
#   padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
#   out = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
#   out = tf.image.resize_images(out, [299, 299], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#   return out
# =============================================================================

# =============================================================================
#   rnd = tf.random_uniform((), 1, 16, dtype=tf.float32)
#   return tf.cond(tf.random_uniform(shape=[1])[0] < 0.5, lambda: input_tensor/rnd, lambda: input_tensor)
# =============================================================================


  rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
  rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  h_rem = FLAGS.image_resize - rnd
  w_rem = FLAGS.image_resize - rnd
  pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
  pad_bottom = h_rem - pad_top
  pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
  pad_right = w_rem - pad_left
  padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
  padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
  out = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)
  out = tf.image.resize_images(out, [299, 299], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  rnd = tf.random_uniform((), 1, 16, dtype=tf.float32)
  return out/rnd

def main(_):
    

  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  
  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.compat.v1.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)
  
    y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
    i = tf.constant(0,float)
    grad = tf.zeros(shape=batch_shape)
    grad2 = tf.zeros(shape=batch_shape)
    x_adv, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad, grad2])
    
    
        # Run computation
    s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
    s2 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
    s3 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
    s4 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
    s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
    s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
    s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))
    s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2'))
  
    with tf.Session() as sess:
      s1.restore(sess, FLAGS.checkpoint_path_inception_v3)
      s2.restore(sess, FLAGS.checkpoint_path_adv_inception_v3)
      s3.restore(sess, FLAGS.checkpoint_path_ens3_adv_inception_v3)
      s4.restore(sess, FLAGS.checkpoint_path_ens4_adv_inception_v3)
      s5.restore(sess, FLAGS.checkpoint_path_inception_v4)
      s6.restore(sess, FLAGS.checkpoint_path_inception_resnet_v2)
      s7.restore(sess, FLAGS.checkpoint_path_ens_adv_inception_resnet_v2)
      s8.restore(sess, FLAGS.checkpoint_path_resnet)
  
      for filenames, images in load_images(FLAGS.input_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        save_images(adv_images, filenames, FLAGS.output_dir)
      

        

        
if __name__ == '__main__':
  tf.compat.v1.app.run()

        



