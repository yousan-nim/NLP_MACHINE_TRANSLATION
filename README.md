# NLP_MACHINE_TRANSLATION


## Sequence to Sequence Learning with Neural Networks Open In Colab
This first tutorial covers the workflow of a PyTorch with torchtext seq2seq project. We'll cover the basics of seq2seq networks using encoder-decoder models,
how to implement these models in PyTorch, and how to use torchtext to do all of the heavy lifting with regards to text processing.
The model itself will be based off an implementation of Sequence to Sequence Learning with Neural Networks, which uses multi-layer LSTMs.



## Packed Padded Sequences, Masking, Inference and BLEU Open In Colab
In this notebook, we will improve the previous model architecture by adding packed padded sequences and masking.
These are two methods commonly used in NLP. Packed padded sequences allow us to only process the non-padded elements of our input sentence with our RNN. 
Masking is used to force the model to ignore certain elements we do not want it to look at, such as attention over padded elements. 
Together, these give us a small performance boost. We also cover a very basic way of using the model for inference,
allowing us to get translations for any sentence we want to give to the model and how we can view the attention values over the source sequence for those translations. 
Finally, we show how to calculate the BLEU metric from our translations.



## Convolutional Sequence to Sequence Learning Open In Colab
We finally move away from RNN based models and implement a fully convolutional model. One of the downsides of RNNs is that they are sequential. That is, 
before a word is processed by the RNN, all previous words must also be processed. Convolutional models can be fully parallelized, which allow them to be trained much quicker.
We will be implementing the Convolutional Sequence to Sequence model, which uses multiple convolutional layers in both the encoder and decoder, with an attention mechanism between them.





## Attention Is All You Need Open In Colab
Continuing with the non-RNN based models, we implement the Transformer model from Attention Is All You Need. 
This model is based soley on attention mechanisms and introduces Multi-Head Attention. 
The encoder and decoder are made of multiple layers, with each layer consisting of Multi-Head Attention and Positionwise Feedforward sublayers. 
This model is currently used in many state-of-the-art sequence-to-sequence and transfer learning tasks.




