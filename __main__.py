import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet,preprocess_input
from dataset_utils import TobiiDataEngine, TfRecordCodec, PcdDataEngine
from model_collections import AttentionModel,PointNet
from model_utils import WeightAnnealing, prediction_loss,build_pretrained_model,CustomizedModelCheckpoint




if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0

    tf.config.experimental.set_virtual_device_configuration(
        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5500)])

    seeds = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1234]

    ######################### training for PointNet ###########################################################

    path_prefix_train = "/home/user_name/"
    path_prefix_test = "/home/user_name/"

    train_sess = ['2021-09-10/0','2021-09-10/1','2021-09-10/2','2021-09-10/3']
    test_sess = ['2021-09-23/0','2021-09-23/1','2021-09-23/2','2021-09-23/3']

    train_sess_list = []
    test_sess_list = []
    for sess in train_sess:
        train_sess_list.append(path_prefix_train+sess+"/train.tfrecords")
    for sess in test_sess:
        test_sess_list.append(path_prefix_test+sess+"/train.tfrecords")

    # a self-defined dataset (TfRecord) loader for the RealSense point clouds
    train_ds = load_dataset(train_sess_list, shuffle_seed=seeds[iter_num],batch_size=64)
    test_ds = load_dataset(test_sess_list, shuffle_seed=seeds[iter_num],batch_size=64)


    for iter_num in range(len(seeds)):
        ckp_rootpath = '/home/user_name/PointNet_{:d}/'.format(iter_num)

        ckp_filepath = ckp_rootpath + 'ckp/weights.{epoch:02d}-{val_loss:.2f}'

        log_filepath = '/home/user_name/log_{:d}.csv'.format(iter_num)

        ckp_cb = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_filepath, monitor='val_loss',
                                                    verbose=0, save_best_only=False, save_weights_only=True,
                                                    mode='auto', save_freq='epoch', options=None)

        log_cb = tf.keras.callbacks.CSVLogger(filename=log_filepath, separator=',', append=False)

        lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, 1e5, 0.998, staircase=True)

        opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)

        if os.path.exists(log_filepath) or os.path.exists(ckp_rootpath):
            raise ValueError('model exists')

        pn = PointNet(is_vanilla=False)

        pn.compile(optimizer=opt, loss=prediction_loss)

        pn.fit(x=train_ds,epochs=10,callbacks=[ckp_cb,log_cb],validation_data=test_ds)

    ######################### training for Gaze-Terrain Network ##################################################

    path_prefix_train = "/home/user_name/"
    path_prefix_test = "/home/user_name/"

    train_sess = ['2021-01-25/1', '2021-01-25/2','2021-01-25/3']
    test_sess = ['2021-03-23/1','2021-03-23/2','2021-03-23/3']

    train_sess_list = []
    test_sess_list = []
    for sess in train_sess:
        train_sess_list.append(path_prefix_train+sess+"/train.tfrecords")
    for sess in test_sess:
        test_sess_list.append(path_prefix_test+sess+"/test.tfrecords") # self-defined dataset (TfRecord) loader


    for iter_num in range(len(seeds)):

        tf.random.set_seed(seeds[iter_num])

        # a self-defined dataset (TfRecord) loader for the TobiiGlasses eye-tracker data
        train_ds = load_dataset(train_sess_list, shuffle_seed=seeds[iter_num],batch_size=50, buffer_size=500)
        test_ds = load_dataset(test_sess_list, shuffle_seed=seeds[iter_num],batch_size=50, buffer_size=500)

        ckp_rootpath ='/home/user_name/GTNet_{:d}/'.format(iter_num)

        ckp_filepath = ckp_rootpath+'ckp/weights.{epoch:02d}-{val_loss:.2f}'

        log_filepath = '/home/user_name/log_{:d}.csv'.format(iter_num)

        ckp_cb = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_filepath, monitor='val_loss',
                                                    verbose=0,save_best_only=False,save_weights_only=True,
                                                    mode='auto', save_freq='epoch',options=None)

        log_cb = tf.keras.callbacks.CSVLogger(filename=log_filepath,separator=',', append=False)

        early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_kl_div', min_delta=1.0e-3, patience=3)

        anl_cb = WeightAnnealing(total_iteration=454*10+1,num_cycle=4,ratio=0.25,beta=1)

        lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, 1e5, 0.998, staircase=True)

        opt = tf.keras.optimizers.Adam(learning_rate=lr_sched)

        if os.path.exists(log_filepath) or os.path.exists(ckp_rootpath):
            raise ValueError('model exists')

        pretrained_model = build_pretrained_model(pretrained_layers=(35,72,76),tunable=True)

        att_model = AttentionModel(pretrained_model,gaze_main_shape=(28,28,256))

        att_model.compile(optimizer=opt, loss=prediction_loss)

        att_model.fit(x=train_ds,epochs=10,callbacks=[ckp_cb,log_cb,anl_cb],validation_data=test_ds)





