# Video Caption Generator

## Overview

This project implements a video caption generator using LSTM (Long Short-Term Memory) networks. It takes input video features and generates descriptive captions for the content of the video. The model has been trained on the MVSD (Multiview Video Dataset) dataset.

## Model Architecture

The model consists of the following components:

### Encoder

The encoder module projects the video features into a different space, which is then passed to the decoder for caption generation. It uses a linear layer for appearance projection.

### Temporal Attention

The temporal attention module focuses on specific parts of the input video features, depending on the previous hidden memory in the decoder and the features at the source side. It calculates attention weights to guide the caption generation process.

### Decoder

The decoder, essentially a language model, generates captions based on the input video features and the attention mechanism. It utilizes LSTM or GRU cells for sequential processing and a linear layer for output prediction.

## Usage

### Model Code

The model code is provided in the following classes:

- `Encoder`: Projects video features into a different space.
- `TemporalAttention`: Calculates temporal attention weights.
- `DecoderRNN`: Generates captions based on input features and attention mechanism.
- `SALSTM`: Implements the entire sequence-to-sequence architecture.

### Training

The `SALSTM` class includes methods for training the model. You can train the model using the provided `train_epoch` and `train_iter` functions.

### Inference

The `GreedyDecoding` method in the `SALSTM` class is used for inference. It generates captions for input video features.

## Dataset

The model has been trained on the MVSD (Multiview Video Dataset). If you require access to this dataset for your own projects, please refer to the original source for licensing and access information.

## Streamlit App

An end-to-end Streamlit web application has been built for easy interaction with the video caption generator model using pretrained model of facebook.

## Contact

For any questions or inquiries regarding the model or the project, you can contact me at:

**Your Name**  
Email: azizulhakim8291@gmail.com

## Acknowledgements

- The model architecture and implementation were inspired by research in the field of video captioning.
- Special thanks to the creators of the MVSD dataset for providing the training data.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
