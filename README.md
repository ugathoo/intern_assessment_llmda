# Next-Word Prediction with LSTM and Attention

A neural network implementation for next-word prediction using LSTM layers with multi-head self-attention mechanism, trained on "The Adventures of Sherlock Holmes" by Arthur Conan Doyle.

## Features

- **Advanced Architecture**: LSTM layers with multi-head self-attention and positional encoding
- **GloVe Embeddings**: Pre-trained word embeddings for better semantic understanding
- **Quality Text Generation**: Advanced sampling techniques including nucleus sampling and repetition penalties
- **Comprehensive Analysis**: Step-by-step generation analysis with probability tracking
- **Model Persistence**: Save/load functionality for trained models

## Installation and Usage
### 1. Setup Environment
Clone the repository:
```bash
# Clone the repository
git clone <your-repo-url>
cd next-word-predictor
```

This project uses modern Python dependency management. Install dependencies using poetry:

```bash
pip install poetry
poetry install
```
Run the following command to start the poetry shell:
```bash
poetry shell
```
If poetry shell is not installed, you may need to stall the plugin manually. See the [poetry plugin shell documentation](https://github.com/python-poetry/poetry-plugin-shell).

### 2.Download GloVe Embeddings

Download the GloVe embeddings file and place it in the project directory:
```bash
# Download GloVe 6B 100d embeddings (862MB)
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
# Keep glove.6B.100d.txt in the project directory
```
### 3. Ensure Additional Requirements Met
- **Python**: 3.8+ required
- **Hardware**: GPU recommended but not required (CPU training supported)

### 4. Run the Model
From inside the poetry shell, run the following command:
```bash
python next_word_predictor.py
```

This will:
- Download "The Adventures of Sherlock Holmes" from Project Gutenberg
- Preprocess and tokenize the text
- Build the LSTM model with attention mechanism
- Train the model
- Generate example texts with analysis
- Save the trained model

### 4. Using Pre-trained Model

If you have a saved model, you can load it and generate text directly. Go to the next_word_predictor.py file's main function and uncomment the line to load the model. Be sure to comment out the lines training the model if you want to use the pre-trained model.

## Architecture

### Model Components

1. **Embedding Layer**: Uses pre-trained GloVe embeddings (100d) with fine-tuning
2. **Positional Encoding**: Custom layer for sequence position awareness
3. **LSTM Layers**: Two stacked LSTM layers (128 units each) with dropout
4. **Multi-Head Attention**: 4-head self-attention mechanism for context modeling
5. **Residual Connections**: Skip connections with layer normalization
6. **Output Layer**: Softmax over vocabulary for next-word prediction

## File Structure
```
next-word-predictor/
├── next_word_predictor.py    # Main implementation
├── README.md                 # This file
├── report.txt               # Detailed technical report
├── best_model.h5           # Saved model weights (after training)
├── next_word_model.keras   # Full saved model (after training)
├── vocabulary.pkl          # Vocabulary mappings (after training)
└── glove.6B.100d.txt      # GloVe embeddings (download required)
```




