import tensorflow as tf
from util import LeakRelu


def bulid_block(in_layer, out_dim, train = True, scope = "block"):
    
    assert in_layer.get_shape()[-1] == out_dim
    
    with tf.variable_scope(scope):
        out_layer = tf.layers.conv2d(in_layer, out_dim, 3, padding = "same", scope = "conv1")
        out_layer = tf.layers.batch_normalization(out_layer, training = train, scope= "bn1")
        out_layer = tf.nn.relu(out_layer)
        out_layer = tf.layers.conv2d(out_layer, out_dim, 3, scope = "conv2")
        out_layer = tf.pad(out_layer, [[0,0],[1,1],[1,1],[0,0]], "CONSTANT")
        
        return tf.nn.relu(out_layer + in_layer)



def ResNet(in_layer, out_dim, num_blocks, train = True, scope = "ResNet"):
    
    with tf.variable_scope(scope):
        h_layer = bulid_block(in_layer, out_dim, train, "block1")
        
        for i in range(1, num_blocks):
            h_layer = bulid_block(h_layer, out_dim, train, "block{}".format(i+1))
            
        return h_layer


def generator_ResNet(input_img, train = True, scope = "gen_Res"):
    img = tf.pad(input_img, [[0,0],[3,3],[3,3],[0,0]])
    with tf.variable_scope(scope):
        h_layer1 = tf.layers.conv2d(img, 32, 7, 1, scope = "conv1")
        h_layer1 = tf.nn.relu(tf.layers.batch_normalization(h_layer1, training = train, scope = "bn1"))
        
        h_layer2 = tf.layers.conv2d(h_layer1, 64, 3, 2, padding = "same", scope = "conv2")
        h_layer2 = tf.nn.relu(tf.layers.batch_normalization(h_layer2, training = train, scope = "bn2"))
        
        h_layer3 = tf.layers.conv2d(h_layer2, 128, 3, 2, padding = "same", scope = "conv3")
        h_layer3 = tf.nn.relu(tf.layers.batch_normalization(h_layer3, training = train, scope = "bn3"))
        
        h_layer4 = ResNet(h_layer3, 128, 6, train = train, scope = "ResNet1")
        
        h_layer5 = tf.layers.conv2d_transpose(h_layer4, 64, 3, 2, padding = "same", scope = "deconv1")
        h_layer5 = tf.nn.relu(tf.layers.batch_normalization(h_layer5, training = train, scope = "bn4"))
        
        h_layer6 = tf.layers.conv2d_transpose(h_layer5, 32, 3, 2, padding = "same", scope = "deconv2")
        h_layer6 = tf.nn.relu(tf.layers.batch_normalization(h_layer6, training = train, scope = "bn5"))
        
        h_layer7 = tf.layers.conv2d(h_layer6, 3, 7, 1, padding = "same", scope = "conv4")
        h_layer7 = tf.nn.tanh(tf.layers.batch_normalization(h_layer7, training = train, scope = "bn6"))
        
        return h_layer7



def discriminator(input_img, train = True, scope = "discriminator"):
    (96,96,3)
    size = input_img.get_shape()[1]/32
    with tf.variable_scope(scope):
        # ->(48,48,64)
        h_layer1 = tf.layers.conv2d(input_img, 64, 3, 2, padding = "same", scope = "conv1")
        h_layer1 = LeakRelu(h_layer1)  
        
        # ->(12,12,128)
        h_layer2 = tf.layers.conv2d(h_layer1, 128, 3, 2, padding = "same", scope = "conv2")
        h_layer2 = tf.layers.max_pooling2d(h_layer2, 3, 2, padding = "same", scope = "max_pool1")
        h_layer2 = tf.layers.batch_normalization(h_layer2, scope = "bn1")
        h_layer2 = LeakRelu(h_layer2)
        
        # ->(3,3,256)
        h_layer3 = tf.layers.conv2d(h_layer2, 256, 3, 2, padding = "same", scope = "conv3")
        h_layer3 = tf.layers.max_pooling2d(h_layer3, 3, 2, padding = "same", scope = "max_pool2")
        h_layer3 = tf.layers.batch_normalization(h_layer3, scope = "bn2")
        h_layer3 = LeakRelu(h_layer3)
        
        flat = tf.reshape(h_layer3, [-1, size*size*256])
        output = tf.layers.dense(flat, 1)
        
        return output

