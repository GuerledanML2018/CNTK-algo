# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:46:46 2018

@author: veylonni
"""


# Import the relevant modules
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import matplotlib.pyplot as plt
import numpy as np
import os
# Import CNTK
import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components



# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        labels_viz = C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
        features   = C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    )), randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)
    
    
def create_deep_model(features):
    with C.layers.default_options(init = C.layers.glorot_uniform()):
        encode = C.element_times(C.constant(1.0/255.0), features)

        for encoding_dim in encoding_dims:
            encode = C.layers.Dense(encoding_dim, activation = C.relu)(encode)

        global encoded_model
        encoded_model= encode

        decode = encode
        for decoding_dim in decoding_dims:
            decode = C.layers.Dense(decoding_dim, activation = C.relu)(decode)

        decode = C.layers.Dense(input_dim, activation = C.sigmoid)(decode)
        return decode
    
    
def train_and_test(reader_train, reader_test, model_func):

    ###############################################
    # Training the model
    ###############################################

    # Instantiate the input and the label variables
    input = C.input_variable(input_dim)
    label = C.input_variable(input_dim)

    # Create the model function
    model = model_func(input)

    # The labels for this network is same as the input MNIST image.
    # Note: Inside the model we are scaling the input to 0-1 range
    # Hence we rescale the label to the same range
    # We show how one can use their custom loss function
    # loss = -(y* log(p)+ (1-y) * log(1-p)) where p = model output and y = target
    # We have normalized the input between 0-1. Hence we scale the target to same range

    target = label/255.0
    loss = -(target * C.log(model) + (1 - target) * C.log(1 - model))
    label_error  = C.classification_error(model, target)

    # training config
    epoch_size = 5760/2        # is half the dataset size
    minibatch_size = 20
    num_sweeps_to_train_with = 5 if isFast else 100
    num_samples_per_sweep = 5760
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) // minibatch_size


    # Instantiate the trainer object to drive the model training
    lr_per_sample = [0.00003]
    lr_schedule = C.learning_parameter_schedule_per_sample(lr_per_sample, epoch_size)

    # Momentum which is applied on every minibatch_size = 64 samples
    momentum_schedule = C.momentum_schedule(0.9126265014311797, minibatch_size)

    # We use a variant of the Adam optimizer which is known to work well on this dataset
    # Feel free to try other optimizers from
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
    learner = C.fsadagrad(model.parameters,
                         lr=lr_schedule, momentum=momentum_schedule)

    # Instantiate the trainer
    progress_printer = C.logging.ProgressPrinter(0)
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    # Map the data streams to the input and labels.
    # Note: for autoencoders input == label
    input_map = {
        input  : reader_train.streams.features,
        label  : reader_train.streams.features
    }

    aggregate_metric = 0
    for i in range(num_minibatches_to_train):
        # Read a mini batch from the training data file
        data = reader_train.next_minibatch(minibatch_size, input_map = input_map)

        # Run the trainer on and perform model training
        trainer.train_minibatch(data)
        samples = trainer.previous_minibatch_sample_count
        aggregate_metric += trainer.previous_minibatch_evaluation_average * samples

    train_error = (aggregate_metric*100.0) / (trainer.total_number_of_samples_seen)
    print("Average training error: {0:0.2f}%".format(train_error))

    #############################################################################
    # Testing the model
    # Note: we use a test file reader to read data different from a training data
    #############################################################################

#    # Test data for trained model
#    test_minibatch_size = 32
#    num_samples = 10000
#    num_minibatches_to_test = num_samples / test_minibatch_size
#    test_result = 0.0
#
#    # Test error metric calculation
    metric_numer    = 0
    metric_denom    = 0
#
#    test_input_map = {
#        input  : reader_test.streams.features,
#        label  : reader_test.streams.features
#    }
#
#    for i in range(0, int(num_minibatches_to_test)):
#
#        # We are loading test data in batches specified by test_minibatch_size
#        # Each data point in the minibatch is a MNIST digit image of 784 dimensions
#        # with one pixel per dimension that we will encode / decode with the
#        # trained model.
#        data = reader_test.next_minibatch(test_minibatch_size,
#                                       input_map = test_input_map)
#
#        # Specify the mapping of input variables in the model to actual
#        # minibatch data to be tested with
#        eval_error = trainer.test_minibatch(data)
#
#        # minibatch data to be trained with
#        metric_numer += np.abs(eval_error * test_minibatch_size)
#        metric_denom += test_minibatch_size
#
#    # Average of evaluation errors of all test minibatches
    test_error = (metric_numer*100.0) / (metric_denom)
#    print("Average test error: {0:0.2f}%".format(test_error))

    return model, train_error, test_error


def print_image_stats(img, text):
    print(text)
    print("Max: {0:.2f}, Median: {1:.2f}, Mean: {2:.2f}, Min: {3:.2f}".format(np.max(img),
                                                                              np.median(img),
                                                                              np.mean(img),
                                                                              np.min(img)))


# Define a helper function to plot a pair of images
def plot_image_pair(img1, text1, img2, text2):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))

    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title(text1)
    axes[0].axis("off")

    axes[1].imshow(img2, cmap="gray")
    axes[1].set_title(text2)
    axes[1].axis("off")
                 
    
if __name__ == '__main__':
    
    isFast = True

    # Récupération des datasets
    try:
        data_dir = os.path.join("DataSets", "MNIST")
        train_file = os.path.join(data_dir, "Train-28x28_cntk_text.txt")
        test_file = os.path.join(data_dir, "Test-28x28_cntk_text.txt")
    except:
        raise ValueError("Please generate the data before executing this script")
    
    # données d'initialisation
    input_dim = 784  # nombre de pixels dans une image
    encoding_dims = [128,64,32]  # TODO : choisir la dimensionnalité
    decoding_dims = [64,128]  # TODO : same
    encoded_model = None  # le modèle qui encode les images
    
    num_label_classes = 10  # ??? : utilisé ou non ? 
    
    # readers
    reader_train = create_reader(train_file, True, input_dim, num_label_classes)
    reader_test = create_reader(test_file, False, input_dim, num_label_classes)
    
    # training
    model, deep_ae_train_error, deep_ae_test_error = train_and_test(reader_train,
                                                                    reader_test,
                                                                    model_func = create_deep_model)
        
    # evaluation
    reader_eval = create_reader(test_file, False, input_dim, num_label_classes)
    eval_minibatch_size = 50
    eval_input_map = { input  : reader_eval.streams.features }    
    eval_data = reader_eval.next_minibatch(eval_minibatch_size,
                                      input_map = eval_input_map)
    img_data = eval_data[input].asarray()
    
    # Select a random image
    np.random.seed(0) 
    idx = np.random.choice(eval_minibatch_size)
    
    # Run the same image as the simple autoencoder through the deep encoder
    orig_image = img_data[idx,:,:]
    decoded_image = model.eval(orig_image)[0]*255
    
    # Print original image
    print_image_stats(orig_image, "Original image statistics:")
    
    # Print decoded image
    print_image_stats(decoded_image, "Decoded image statistics:")
    
    # Plot the original and the decoded image
    img1 = orig_image.reshape(28,28)
    text1 = 'Original image'    
    img2 = decoded_image.reshape(28,28)
    text2 = 'Decoded image'    
    plot_image_pair(img1, text1, img2, text2)
    
    # Exemple d'encodage d'une image
    img = orig_image
    img_encoded =  encoded_model.eval([img])
    
    
    
    
    
    
    