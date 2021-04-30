"""
-- directory
   -- LeNet5.py
   -- caller.py
   -- utils
      -- ##### basic_utils.py
      -- ##### cp_utils.py
      -- ##### dataset_utils.py
      -- ##### learning_env_setting.py
      -- ##### train_validation_test.py

   -- train1
      -- confusion_matrix
      -- model
      -- losses_accs.npz
      -- losses_accs_visualization.png
      -- test_result.txt
"""

import os
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

from utils.learning_env_setting import argparser, dir_setting, continue_setting, get_classification_metrics
from utils.dataset_utils import load_processing_mnist, load_processing_cifar10
from utils.train_validation_test import train, validation, test
from utils.cp_utils import save_metrics_model, metrics_visualizer
from utils.basic_utils import resetter, training_reporter

from models import LeNet5
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

'''===== Learning Setting ===='''
CONTINUE_LEARNING = False
exp_name = 'learning_rate'

train_ratio = 0.8
train_batch_size, test_batch_size = 32, 128

epochs = 50
save_period = 10
learning_rate = 0.01

exp_idx, epochs, learning_rate, training_batch_size, activation = argparser(epochs, learning_rate, train_batch_size)

exp_name = 'exp' + str(exp_idx) + '_' + exp_name + '_LeNet5'
model = LeNet5(activation=activation)
optimizer = SGD(learning_rate=learning_rate)
'''===== Learning Setting ===='''

loss_object = SparseCategoricalCrossentropy()
path_dict = dir_setting(exp_name, CONTINUE_LEARNING)
model, losses_accs, start_epoch = continue_setting(CONTINUE_LEARNING, path_dict, model)
train_ds, validation_ds, test_ds = load_processing_mnist(train_ratio, train_batch_size, test_batch_size)

metric_objects = get_classification_metrics()

for epoch in range(start_epoch, epochs):
    train(train_ds, model, loss_object, optimizer, metric_objects)
    validation(validation_ds, model, loss_object, metric_objects)

    training_reporter(epoch, losses_accs, metric_objects)
    save_metrics_model(epoch, model, losses_accs, path_dict, save_period)

    metrics_visualizer(losses_accs, path_dict['cp_path'])
    resetter(metric_objects)

test(test_ds, model, loss_object, metric_objects, path_dict)
