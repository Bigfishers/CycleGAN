
# coding: utf-8

# In[1]:


import tensorflow as tf
from model import generator_ResNet, discriminator
import data
from data import input_data


# In[2]:


import os
import cv2
import numpy as np
from util import save_img


# In[3]:


def train_model(data_img, width, height, depth, batch_size=100, num_epoch = 5000):
    input_A = tf.placeholder(tf.float32, [None, width, height, depth], name="input_A") #X
    input_B = tf.placeholder(tf.float32, [None, width, height, depth], name="input_B") #Y
    
    smooth = 0.9
    lr = 0.001
    
    with tf.variable_scope("model") as scope:
        fake_A = generator_ResNet(input_B, scope="G") #G(y) g_B B2A
        fake_B = generator_ResNet(input_A, scope="F") #F(x) g_A A2B
        d_real_A = discriminator(input_A, scope="Dx") #Dx(x) d_A
        d_real_B = discriminator(input_B, scope="Dy") #Dy(y) d_B
        
        scope.reuse_variables()
        cyc_A = generator_ResNet(fake_B, scope="G") #G(F(x))
        cyc_B = generator_ResNet(fake_A, scope="F") #F(G(y))
        d_fake_A = discriminator(fake_A, scope="Dx") #Dx(G(y))
        d_fake_B = discriminator(fake_B, scope="Dy") #Dy(F(x))
        
    #loss    
    d_loss_fake_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_B,
                                                                           labels = tf.zeros_like(d_fake_B)))
                                                                            
    d_loss_real_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_real_B,
                                                                           labels = tf.ones_like(d_real_B)* smooth))
    d_loss_fake_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_A,
                                                                           labels = tf.zeros_like(d_fake_A)))
    d_loss_real_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_real_A,
                                                                           labels = tf.ones_like(d_real_A)* smooth))
    d_loss_A = d_loss_fake_A + d_loss_real_A
    d_loss_B = d_loss_fake_B + d_loss_real_B
    
    cycle_loss = tf.reduce_mean(tf.abs(cyc_A - input_A))+tf.reduce_mean(tf.abs(cyc_B - input_B))
    g_loss_A = cycle_loss * 10 + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_B, 
                                                                                       labels = tf.ones_like(d_fake_B) *smooth))
    g_loss_B = cycle_loss * 10 + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake_A,
                                                                                       lables = tf.ones_like(d_fake_A) *smooth))
    
    train_vars = tf.trainable_variables()
    g_A_vars = [var for var in train_vars if "F" in var.name]
    g_B_vars = [var for var in train_vars if "G" in var.name]
    d_A_vars = [var for var in train_vars if "Dx" in var.name]
    d_B_vars = [var for var in train_vars if "Dy" in var.name]
    
    #optimizer
    d_A_opt = tf.train.AdamOptimizer(lr).minimize(d_loss_A, var_list=d_A_vars)
    d_B_opt = tf.train.AdamOptimizer(lr).minimize(d_loss_B, var_list=d_B_vars)
    g_A_opt = tf.train.AdamOptimizer(lr).minimize(g_loss_A, var_list=g_A_vars)
    g_B_opt = tf.train.AdamOptimizer(lr).minimize(g_loss_B, var_list=g_B_vars)
    
    #summary
    d_A_summ = tf.summary.scalar("discriminator_A_loss", d_loss_A)
    d_B_summ = tf.summary.scalar("discriminator_B_loss", d_loss_B)
    g_A_summ = tf.summary.scalar("generator_A2B_loss", g_loss_A)
    g_B_summ = tf.sumaary.scalar("generator_B2A_loss", g_loss_B)
    
    if not os.path.exists("output"):
        os.mkdir("output")
    summary_writer = tf.summary.FileWriter("output/summary")
    
    #init
    init = tf.variables_initializer()
    saver = tf.train.Saver()
    
    #train
    with tf.device('/gpu:0'):
        with tf.Session() as sess:
            
            sess.run(init)
            
            if not os.path.exists("output/checkpoints"):
                os.mkdir("output/checkpoints")
            if not os.path.exists("output/samples"):
                os.mkdir("output/samples")
                
            #loop
            for epoch in range(num_epoch):
                print("Start epoch :{}".format(epoch/100))
                    
                for _ in range(data_img.num_examples/batch_size):
                    temp_A = data_img.A.next_batch()
                    temp_B = data_img.B.next_batch()
                    #G_A
                    _, summary_str = sess.run([g_A_opt, g_A_summ], feed_dict= {input_A:temp_A, input_B:temp_B})
                    summary_writer.add_summary(summary_str)
                    #D_B
                    _, summary_str = sess.run([d_B_opt, d_B_summ], feed_dict= {input_A:temp_A, input_B:temp_B})
                    summary_writer.add_summary(summary_str)
                    #G_B
                    _, summary_str = sess.run([g_B_opt, g_B_summ], feed_dict= {input_A:temp_A, input_B:temp_B})
                    summary_writer.add_summary(summary_str)
                    #D_A
                    _, summary_str = sess.run([d_A_opt, d_B_summ], feed_dict= {input_A:temp_A, input_B:temp_B})
                    summary_writer.add_summary(summary_str)
                    
                if (epoch+1)% 100 ==0:
                    saver.save(sess, "output/checkpoints/cycgan", global_step= epoch+1)
                    save_img(sess, fake_B, data_img, (epoch+1)/100)
                    
            summary_writer.add_graph(sess.graph)
                    
    

