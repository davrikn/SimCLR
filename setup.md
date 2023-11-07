# Setting up

The original setup used pipenv which seemed to work very poorly.
Instead we've opted to go with conda

To setup run

```
conda create -n SimCLR python=3.8
conda activate SimCLR
python -m pip install -r requirements.txt
# Change to a version matching your CUDA install
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

Once everything is installed you can run the project in the following order

```
python 0_train.py # Trains encoder with SimCLR 
python embedd_all.py # Embeds the dataset with the encoder and saves it in numpy format
python train_head.py # Trains a prediction head on the embedded representations
python predict.py # Runs predictions on the embedded representations, and prints the accuracy
```