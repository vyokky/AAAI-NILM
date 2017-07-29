############################################################
# This code is to train a neural network to perform energy disaggregation, 
# i.e., given a sequence of electricity mains reading, the algorithm
# separates the mains into appliances.
#
# Inputs: mains windows with size=(nosOfSamples,1,windowlength)
# Targets: class labels with size=(nosOfSamples,)
#
# In reality all the electricity readings are integers;
# we treat the problem as a classification problem
# where the targets are class labels; this is reasonable 
# since each appliance has a finite number of states,
# e.g., OFF, ON, and intermediate states. 
#
# Since the number of states is greater than 3,000, the states
# are compressed by using quantization. 
# We used mu-law where mu=255.
#
# This code is written by Chaoyun Zhang and Mingjun Zhong.
############################################################

import NetFlowExt as nf
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import DataProvider
import argparse

# only one GPU is visible to current task.
# CUDA_VISIBLE_DEVICES=1 

application = 'kettle'
datadir = '/home/vyokky/aaai/' + application
save_path = './cnn_'+application+'_s2s'
batchsize = 5000
epoch = 100

saver = 1

# def remove_space(string):
#     return string.replace(" ","")

# def get_arguments():
#     parser = argparse.ArgumentParser(description='Train a neural network\
#                                      for energy disaggregation - \
#                                      network input = mains window; \
#                                      network target = the states of \
#                                      the target appliance.')
#     parser.add_argument('--appliance_name',
#                         type=remove_space,
#                         default='dish washer',
#                         help='the name of target appliance')
#     parser.add_argument('--datadir',
#                         type=str,
#                         default='../../data/uk-dale/trainingdata/small/',
#                         help='this is the directory of the training samples')
#     parser.add_argument('--batchsize',
#                         type=int,
#                         default=1000,
#                         help='The batch size of training examples')
#     parser.add_argument('--n_epoch',
#                         type=int,
#                         default=50,
#                         help='The number of epoches.')
#     parser.add_argument('--save_model',
#                         type=int,
#                         default=-1,
#                         help='Save the learnt model: \
#                             0 -- not to save the learnt model parameters;\
#                             n (n>0) -- to save the model params every n steps;\
#                             -1 -- only save the learnt model params \
#                                     at the end of training.')
#     return parser.parse_args()

# Units:
# windowlength: number of data points
# on_power_threshold,max_on_power: power
#params_appliance = {'kettle':{'windowlength':129,
#                              'on_power_threshold':2000,
#                              'max_on_power':3948},
#                    'microwave':{'windowlength':129,
#                              'on_power_threshold':200,
#                              'max_on_power':3138},
#                    'fridge':{'windowlength':299,
#                              'on_power_threshold':50,
#                              'max_on_power':2572},
#                    'dishwasher':{'windowlength':599,
#                              'on_power_threshold':10,
#                              'max_on_power':3230},
#                    'washingmachine':{'windowlength':599,
#                              'on_power_threshold':20,
#                              'max_on_power':3962}}

params_appliance = {'kettle':{'windowlength':129,
                              'on_power_threshold':2000,
                              'max_on_power':3998,
                             'mean':700,
                             'std':1000,
                             's2s_length':128},
                    'microwave':{'windowlength':129,
                              'on_power_threshold':200,
                              'max_on_power':3969,
                                'mean':500,
                                'std':800,
                                's2s_length':128},
                    'fridge':{'windowlength':299,
                              'on_power_threshold':50,
                              'max_on_power':3323,
                             'mean':200,
                             'std':400,
                             's2s_length':512},
                    'dishwasher':{'windowlength':599,
                              'on_power_threshold':10,
                              'max_on_power':3964,
                                  'mean':700,
                                  'std':1000,
                                  's2s_length':1536},
                    'washingmachine':{'windowlength':599,
                              'on_power_threshold':20,
                              'max_on_power':3999,
                                      'mean':400,
                                      'std':700,
                                      's2s_length':2000}}

# args = get_arguments()
# print args.appliance_name
appliance_name = application


def load_dataset():
    tra_x = datadir+'/'+application + '_train_x' #save path for mains
    val_x = datadir+'/'+application + '_val_x'

    tra_y = datadir+'/'+application + '_train_y' #save path for target
    val_y = datadir+'/'+application + '_val_y'

    tra_set_x = np.load(tra_x+'.npy')
    tra_set_y = np.load(tra_y+'.npy')
    val_set_x = np.load(val_x+'.npy')
    val_set_y = np.load(val_y+'.npy')

    print('training set:', tra_set_x.shape, tra_set_y.shape)
    print('validation set:', val_set_x.shape, val_set_y.shape)

    return tra_set_x, tra_set_y, val_set_x,  val_set_y

# load the data set
tra_set_x, tra_set_y, val_set_x,  val_set_y = load_dataset()

# get the window length of the training examples
windowlength = params_appliance[application]['windowlength']

sess = tf.InteractiveSession()


offset = int(0.5*(params_appliance[application]['windowlength']-1.0))

tra_kwag = {
    'inputs': tra_set_x,
    'targets': tra_set_y,
    'flatten':False}

val_kwag = {
    'inputs': val_set_x,
    'targets': val_set_y,
    'flatten':False}

# tra_provider = DataProvider.DoubleSourceSlider(batchsize = batchsize,
#                                                  shuffle = True, offset=offset)
# val_provider = DataProvider.DoubleSourceSlider(batchsize = 5000,
#                                                  shuffle = False, offset=offset)

tra_provider = DataProvider.S2S_Slider(batchsize = batchsize,
                                                 shuffle = True, length = params_appliance[application]['windowlength'])
val_provider = DataProvider.S2S_Slider(batchsize = 5000,
                                                 shuffle = False, length = params_appliance[application]['windowlength'])


x = tf.placeholder(tf.float32,
                   shape=[None, windowlength],
                   name='x')
y_ = tf.placeholder(tf.float32, shape=[None, windowlength], name='y_')

network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.ReshapeLayer(network,
                                 shape=(-1, windowlength, 1, 1))
network = tl.layers.Conv2dLayer(network,
                                act = tf.nn.relu,
                                shape = [10, 1, 1, 6],
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                name = 'cnn1')
network = tl.layers.Conv2dLayer(network,
                                act = tf.nn.relu,
                                shape = [8, 1, 6, 10],
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                name = 'cnn2')
network = tl.layers.Conv2dLayer(network,
                                act = tf.nn.relu,
                                shape = [6, 1, 10, 20],
                                strides=[1, 1, 1, 1],
                                padding='SAME',
                                name = 'cnn3')
network = tl.layers.FlattenLayer(network,
                                 name='flatten')
network = tl.layers.DenseLayer(network,
                               n_units=1024,
                               act = tf.nn.relu,
                               name='dense2')
network = tl.layers.DenseLayer(network,
                               n_units=params_appliance[application]['windowlength'],
                               act = tf.identity,
                               name='output_layer')


y = network.outputs
sess.run(tf.global_variables_initializer())
# params = tl.files.load_npz(path='', name='cnn_lstm_model.npz')
# tl.files.assign_params(sess, params, network)
# print 'set sucessful'

# save_path = './cnn'+appliance_name+'_pointnet_model'
'

nf.customfit(sess = sess,
             network = network,
             cost = cost,
             train_op = train_op,
             tra_provider = tra_provider,
             x = x,
             y_ = y_,
             acc=None,
             n_epoch= epoch,
             print_freq=1,
             val_provider=val_provider,
             save_model=saver,
             tra_kwag=tra_kwag,
             val_kwag=val_kwag ,
             save_path=save_path,
             epoch_identifier=None,
             earlystopping=True,
             min_epoch=1,
             patience=10)

sess.close()

