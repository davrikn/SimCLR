import numpy as np
from src.model import ContrastiveEncoder
from utils.load_data import Loader
from utils.arguments import get_config, get_arguments


def transform_data(data_loader, check_samples=False):
    # Training dataset
    train_loader = data_loader.train_loader
    # Validation dataset
    test_loader = data_loader.test_loader
    # Show samples from training set
    # Get training and test data. Iterator returns a tuple of 3 variables. Pick the first ones as Xtrain, and Xtest
    ((Xtrain, _), ytrain) = next(iter(train_loader))
    ((Xtest, _), ytest)  = next(iter(test_loader))
    # Print informative message as a sanity check
    print(f"Number of samples in training set: {Xtrain.shape}")
    # Make it a 2D array of batch_size x remaining dimension so that we can use it with PCA for baseline performance
    Xtrain2D, Xtest2D = Xtrain.view(Xtrain.shape[0], -1), Xtest.view(Xtest.shape[0], -1)
    # Return arrays
    return Xtrain2D, ytrain, Xtest2D, ytest

config = get_config(get_arguments())
config["dataset"] = "CIFAR10"
config["img_size"] = 32
config["batch_size"] = 1000

dataloader = Loader(config, train=False)

t, y, test, ytest = transform_data(dataloader)

encoder = ContrastiveEncoder(config)
encoder.load_models()

x, y = encoder.predict(dataloader.train_loader)

x = np.array(x)
y = np.array(y)

np.save('./x_CIFAR_rnet50.npy', x)
np.save('./y_CIFAR_rnet50.npy', y)
