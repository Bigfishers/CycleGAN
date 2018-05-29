import tensorflow as tf
import numpy as np
import cv2



def LeakRelu(x, alpha = 0.01):
    return tf.maximum(x, x*alpha)




def save_img(sess, fake_B, datasets, label, max_limit, min_limit = 0):
    rand = np.random.randint(min_limit, max_limit)
    input_img = datasets.A.image[rand].resize([1,96,96,3])
    output_img = sess.run([fake_B], feed_dict={input_A: input_img})
    input_img.resize([96,96,3])
    output_img.resize([96,96,3])
    cv2.imwrite("output/samples/A_"+ str(label)+".jpg", input_img.astype(np.uint8))
    cv2.imwrite("output/samples/B_"+ str(label)+".jpg", output_img.astype(np.uint8))
    return output_img

