## Story Understanding with Deep Bi-Directional Recurrent Neural Networks

This project was done in a team of 4 in the scope of the course Statistical Natural Language Processing at University College London. The aim was to design and implement a machine learning system that is capable of correctly ordering the sentences of a story - i.e. imitating human "understanding of stories". We achieved top-tier accuracy compared to competing teams, predicting 4 out of 5 sentences per story correctly on average, using the setup described in the subsequent paragraphs.
</b>

### Overview

Our implementation is based on a (Tensorflow 0.7.1) Multi-layer Bi-directional RNN, using LSTM, to produce Sentence Embeddings, followed by a feed forward ReLU layer. The entire code, excluding the proprietary data, can be found in the iPython notebook "full_code.ipynb".

### Preprocessing

We implemented a complex set of rules that ensures uniform word representation across the corpus. The tokenization converts abbreviations ("won't") to their original form ("will not"), makes words lower case, and removes punctuation, numbers, and short words (< 2 characters). After the tokenisation, a common, freely available algorithmic stemmer (Porter Stemmer) converts each word into its stem. Then, in order to adapt the shape of input values to our model, stories were split up into single sentences in which every word was represented as a vector.

### Model

The excellent performance of Recurrent Neural Networks in NLP tasks motivated us to implement this type of Neural Network, precisely a multi-layered bidirectional RNN. We introduced multiple layers of forward- and backward-directed LSTM cells to realise the bi-directional passing of values. `DropoutWrapper` with optimized dropout (keep-) probability was used on all layers for regularisation. Furthermore, we found that including a small L2 regularization factor on weights (excl. biases), randomising batching, and learning rate decay improved our results by preventing overfitting. The generated sentence embeddings are then reshaped and passed on to a single fully connected ReLU layer with batch normalisation, before computing logits and applying the softmax function to obtain posterior probabilities for each class. The loss comprises of L2 regularization and the probability error (sparse softmax cross entropy) and is optimized using ‘adam’ (found to be performing best) with gradient clipping. Posterior probabilities are eventually converted to class labels by applying `argmax`.

### Hyperpameter tuning

A script that trains the model using random values for hyperparameters (within predefined ranges) was used for tuning (Appendix 4). In order to define the ranges of the hyperparameters, we created a small dataset of hyperparameters and peak `dev_accuracy` for each hyperparameter. Furthermore, early stopping was implemented to minimize train time and save the model that performed best.

Additionally, to overcome volatility in development set accuracy, we developed source code that computes a moving average of regularly computed development accuracies (ten times per epoch) in order to define a robust amount of training epochs for the final model (Appendix). An example plot is shown below.

### Techniques that were implemented but were found to be not beneficial

* GRU Cells - LSTMs were preferred as their use led to higher accuracy
* Trigram Kneser-Ney Language model to substitue OOVs (Appendix) - given the time constraints, and the required time for the training of the LM, it was not used.
* GloVe pretrained word embeddings (Appendix) 
* Dropout implemented, but best dropout keep probability was 1 (i.e. no dropout).
