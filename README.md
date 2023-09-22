## Image Caption Generator

[![Issues](https://img.shields.io/github/issues/dabasajay/Image-Caption-Generator.svg?color=%231155cc)](https://github.com/dabasajay/Image-Caption-Generator/issues)
[![Forks](https://img.shields.io/github/forks/dabasajay/Image-Caption-Generator.svg?color=%231155cc)](https://github.com/dabasajay/Image-Caption-Generator/network)
[![Stars](https://img.shields.io/github/stars/dabasajay/Image-Caption-Generator.svg?color=%231155cc)](https://github.com/dabasajay/Image-Caption-Generator/stargazers)
[![Ajay Dabas](https://img.shields.io/badge/Ajay-Dabas-825ee4.svg)](https://dabasajay.github.io/)

A neural network to generate captions for an image using CNN and RNN with BEAM Search.

<p align="center">
  <strong>Examples</strong>
</p>

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1600/1*6BFOIdSHlk24Z3DFEakvnQ.png" width="85%" title="Example of Image Captioning" alt="Example of Image Captioning">
</p>

<p align="center">
	Image Credits : <a href="https://towardsdatascience.com/image-captioning-in-deep-learning-9cd23fb4d8d2">Towardsdatascience</a>
</p>

## 1. Requirements

Recommended System Requirements to train model.

<ul type="square">
	<li>A good CPU and a GPU with atleast 8GB memory</li>
	<li>Atleast 8GB of RAM</li>
	<li>Active internet connection so that keras can download inceptionv3/vgg16 model weights</li>
</ul>

Required libraries for Python along with their version numbers used while making & testing of this project

<ul type="square">
	<li>Python - 3.6.7</li>
	<li>Numpy - 1.16.4</li>
	<li>Tensorflow - 1.13.1</li>
	<li>Keras - 2.2.4</li>
	<li>nltk - 3.2.5</li>
	<li>PIL - 4.3.0</li>
	<li>Matplotlib - 3.0.3</li>
	<li>tqdm - 4.28.1</li>
</ul>

<ul type="square">
	<li><a href="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip">Flickr8k_Dataset</a></li>
	<li><a href="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip">Flickr8k_text</a></li>
</ul>

<strong>Important:</strong> After downloading the dataset, put the reqired files in train_val_data folder


## 4. Configurations (config.py)

**config**

1. **`images_path`** :- Folder path containing flickr dataset images
2. `train_data_path` :- .txt file path containing images ids for training
3. `val_data_path` :- .txt file path containing imgage ids for validation
4. `captions_path` :- .txt file path containing captions
5. `tokenizer_path` :- path for saving tokenizer
6. `model_data_path` :- path for saving files related to model
7. **`model_load_path`** :- path for loading trained model
8. **`num_of_epochs`** :- Number of epochs
9. **`max_length`** :- Maximum length of captions. This is set manually after training of model and required for test.py
10. **`batch_size`** :- Batch size for training (larger will consume more GPU & CPU memory)
11. **`beam_search_k`** :- BEAM search parameter which tells the algorithm how many words to consider at a time.
11. `test_data_path` :- Folder path containing images for testing/inference
12. **`model_type`** :- CNN Model type to use -> inceptionv3 or vgg16
13. **`random_seed`** :- Random seed for reproducibility of results

**rnnConfig**

1. **`embedding_size`** :- Embedding size used in Decoder(RNN) Model
2. **`LSTM_units`** :- Number of LSTM units in Decoder(RNN) Model
3. **`dense_units`** :- Number of Dense units in Decoder(RNN) Model
4. **`dropout`** :- Dropout probability used in Dropout layer in Decoder(RNN) Model

## 5. Frequently encountered problems

- **Out of memory issue**:
  - Try reducing `batch_size`
- **Results differ everytime I run script**:
  - Due to stochastic nature of these algoritms, results *may* differ slightly everytime. Even though I did set random seed to make results reproducible, results *may* differ slightly.
- **Results aren't very great using beam search compared to argmax**:
  - Try higher `k` in BEAM search using `beam_search_k` parameter in config. Note that higher `k` will improve results but it'll also increase inference time significantly.




