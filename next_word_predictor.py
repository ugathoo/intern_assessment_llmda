#!/usr/bin/env python3
# next_word_predictor.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Embedding, Dropout,
    Input, Activation, Add, LayerNormalization,
    GlobalAveragePooling1D, Bidirectional
)
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import requests 
import re
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import logging

BATCH_SIZE = 32  # Reduced for better gradient updates

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_glove_file(glove_path: str):
    """
    Loads GloVe embeddings into a dict: word -> np.array of size (embedding_dim,)
    """
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split()
            word = parts[0]
            vec = np.asarray(parts[1:], dtype='float32')
            embeddings_index[word] = vec
    logger.info(f"Loaded {len(embeddings_index)} word vectors from {glove_path}")
    return embeddings_index

class NextWordPredictor:
    def __init__(self, sequence_length=15, lstm_units=128, vocab_size=1500):
        """
        Initialize the Next Word Predictor
        
        Args:
            sequence_length: Length of input sequences (optimized for 114k tokens)
            lstm_units: Number of LSTM units (reduced for small dataset)
            vocab_size: Maximum vocabulary size (appropriate for dataset size)
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.vocab_size = vocab_size
        self.model = None
        self.word_to_idx = {}
        self.idx_to_word = {}
        # for GloVe
        self.embedding_matrix = None
        self.embedding_dim = None

    def download_text(self, url="https://www.gutenberg.org/files/1661/1661-0.txt"):
        logger.info("Downloading text from Project Gutenberg...")
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def clean_text(self, text):
        logger.info("Cleaning text...")
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker   = "*** END OF THE PROJECT GUTENBERG EBOOK"
        start_idx = text.find(start_marker)
        end_idx   = text.find(end_marker)
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx + len(start_marker) : end_idx]
        
        # More aggressive cleaning but preserve sentence structure
        text = re.sub(r'\r\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        # Keep more punctuation for better sentence boundaries
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"\(\)]', '', text)
        text = re.sub(r'_{2,}', '', text)
        text = re.sub(r'\*{2,}', '', text)
        text = re.sub(r'CHAPTER\s+[IVX]+', '', text)
        text = text.lower().strip()
        
        # Better sentence filtering
        sentences = re.split(r'[.!?]+', text)
        return '. '.join(sentences)

    def tokenize_text(self, text):
        logger.info("Tokenizing text...")
        # Better tokenization that preserves word boundaries
        words = re.findall(r'\b\w+\b|[.!?,:;]', text)
        words = [w for w in words if len(w) > 0]
        
        counts = Counter(words)
        # Keep more common words for better coverage
        most_common = counts.most_common(self.vocab_size - 1)
        
        self.word_to_idx = {'<UNK>': 0}
        self.idx_to_word = {0: '<UNK>'}
        
        for idx, (w, _) in enumerate(most_common, 1):
            self.word_to_idx[w] = idx
            self.idx_to_word[idx] = w
        
        tokenized = [self.word_to_idx.get(w, 0) for w in words]
        coverage = sum(1 for w in words if w in self.word_to_idx) / len(words) * 100
        logger.info(f"Vocab size: {len(self.word_to_idx)}, tokens: {len(tokenized)}, coverage: {coverage:.2f}%")
        return tokenized

    def create_sequences(self, tokenized_text):
        """
        Many-to-many sequence creation: each position predicts the next word
        """
        logger.info("Creating many-to-many sequences...")
        seqs, targets = [], []
        
        # Use aggressive stride to get enough samples from small dataset
        stride = max(1, self.sequence_length // 10)
        
        for i in range(0, len(tokenized_text) - self.sequence_length, stride):
            seq = tokenized_text[i:i + self.sequence_length]
            target = tokenized_text[i + 1:i + self.sequence_length + 1]  # Shifted by 1
            
            # Skip sequences with too many unknown tokens (relaxed for small dataset)
            if seq.count(0) < self.sequence_length * 0.4:
                seqs.append(seq)
                targets.append(target)
        
        X = np.array(seqs)
        y = np.array(targets)
        # Convert to categorical for each position
        y = tf.keras.utils.to_categorical(y, num_classes=len(self.word_to_idx))
        
        logger.info(f"  → {len(seqs)} samples, input shape: {X.shape}, target shape: {y.shape}")
        return X, y

    def load_glove_embeddings(self, glove_path: str, embedding_dim=100):
        glove_index = load_glove_file(glove_path)
        self.embedding_dim = embedding_dim
        vocab_size = len(self.word_to_idx)
        
        # Initialize with small random values instead of zeros
        self.embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim)).astype('float32')
        
        hits = 0
        for word, idx in self.word_to_idx.items():
            vec = glove_index.get(word)
            if vec is not None:
                self.embedding_matrix[idx] = vec
                hits += 1
        
        logger.info(f"Built embedding matrix with shape {self.embedding_matrix.shape}")
        logger.info(f"GloVe hits: {hits}/{vocab_size} ({hits/vocab_size*100:.1f}%)")

    def build_mto_model(self, hidden_size=None, optimizer=None):
        if hidden_size is None:
            hidden_size = self.lstm_units
        if optimizer is None:
            optimizer = Adam(learning_rate=0.001)

        inputs = Input(shape=(self.sequence_length,))

        # Embedding with pre-trained GloVe
        x = Embedding(
            input_dim=len(self.word_to_idx),
            output_dim=self.embedding_dim,
            embeddings_initializer=Constant(self.embedding_matrix),
            trainable=True,
            mask_zero=True
        )(inputs)

        # Single LSTM layer (return sequences for many-to-many)
        x = LSTM(hidden_size, return_sequences=True)(x)

        # Single attention layer
        x = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=hidden_size // 4
        )(x, x)

        # Output layer for each position
        outputs = Dense(len(self.word_to_idx), activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        self.model = model
        logger.info("Built simple many-to-many LSTM + attention model")
        print(f"→ Model params: {model.count_params():,}")
        return model

    def train_model(self, X, y, epochs, validation_split=0.2):
        logger.info("Training model...")
        monitor_val = 'val_accuracy'
        
        # Adjusted callbacks for small dataset
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=8,  # Reduced patience for small dataset
            restore_best_weights=True,
            verbose=1
        )
        
        ckpt = ModelCheckpoint(
            'best_model.h5', 
            monitor=monitor_val,
            save_best_only=True, 
            mode='max',
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5,  # More aggressive LR reduction
            patience=4,  # Shorter patience
            min_lr=1e-6, 
            verbose=1
        )
        
        # Use class weights to handle imbalanced vocabulary
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            validation_split=validation_split,
            callbacks=[early_stop, ckpt, reduce_lr],
            verbose=1
        )
        return history

    def calculate_perplexity(self, X, y):
        logger.info("Calculating perplexity...")
        ce = 0.0
        n = 0
        for i in range(0, len(X), BATCH_SIZE):
            bx = X[i:i+BATCH_SIZE]
            by = y[i:i+BATCH_SIZE]
            preds = self.model.predict(bx, verbose=0)
            
            # For many-to-many, we have predictions for each timestep
            for j in range(len(by)):
                for k in range(by.shape[1]):  # Each timestep
                    true_idx = np.argmax(by[j, k])
                    p = preds[j, k, true_idx]
                    ce += -np.log(max(p, 1e-10))
                    n += 1
        
        ce /= n
        perp = np.exp(ce)
        logger.info(f"Perplexity: {perp:.2f}")
        return perp

    def generate_text(self, seed_text, max_phrase_len=50, temperature=1.0):
        """
        Generate text using many-to-many model
        """
        model = self.model
        seed_words = re.findall(r'\b\w+\b|[.!?,:;]', seed_text.lower())
        seed_tokens = [self.word_to_idx.get(w, 0) for w in seed_words]
        
        # Pad sequence if too short
        if len(seed_tokens) < self.sequence_length:
            seed_tokens = [0] * (self.sequence_length - len(seed_tokens)) + seed_tokens

        generated = seed_text
        steps = []
        
        for _ in range(max_phrase_len):
            inp = np.array([seed_tokens[-self.sequence_length:]])
            preds = model.predict(inp, verbose=0)
            
            # Get prediction for last timestep
            preds = preds[0, -1, :]
            
            # Apply temperature
            if temperature > 0:
                preds = np.log(preds + 1e-10) / temperature
                preds = np.exp(preds) / np.sum(np.exp(preds))

            # Get top 5 predictions
            top5_idx = np.argsort(preds)[-5:][::-1]
            top5 = [(self.idx_to_word.get(i, '<UNK>'), float(preds[i])) for i in top5_idx]
            steps.append(top5)

            # Sample next word
            if temperature > 0:
                top_k = min(15, len(preds))
                tk_idx = np.argsort(preds)[-top_k:]
                tk_p = preds[tk_idx] / np.sum(preds[tk_idx])
                next_idx = np.random.choice(tk_idx, p=tk_p)
            else:
                next_idx = np.argmax(preds)

            next_word = self.idx_to_word.get(next_idx, '<UNK>')
            
            # Stop conditions
            if next_word in ('.', '!', '?', '<UNK>'):
                if next_word in '.!?':
                    generated += next_word
                break

            generated += " " + next_word
            seed_tokens.append(next_idx)

        return generated, steps

    def save_model(self, filepath="next_word_model.keras"):
        self.model.save(filepath)
        with open("vocabulary.pkl", "wb") as f:
            pickle.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'sequence_length': self.sequence_length
            }, f)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath="next_word_model.keras"):
        self.model = tf.keras.models.load_model(filepath)
        with open("vocabulary.pkl", "rb") as f:
            data = pickle.load(f)
            self.word_to_idx = data['word_to_idx']
            self.idx_to_word = data['idx_to_word']
            self.sequence_length = data['sequence_length']
        logger.info(f"Model loaded from {filepath}")

    def plot_training_history(self, history):
        """Plot training history"""
        try:
            h = history.history
            acc_key = 'accuracy' if 'accuracy' in h else 'categorical_accuracy'
            val_key = f"val_{acc_key}"
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(h['loss'], label='Train Loss')
            plt.plot(h['val_loss'], label='Val Loss')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.yscale('log')
            
            plt.subplot(1, 3, 2)
            plt.plot(h[acc_key], label='Train Acc')
            plt.plot(h[val_key], label='Val Acc')
            plt.title('Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            if 'top_k_categorical_accuracy' in h:
                plt.plot(h['top_k_categorical_accuracy'], label='Train Top-K Acc')
                plt.plot(h['val_top_k_categorical_accuracy'], label='Val Top-K Acc')
                plt.title('Top-K Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Top-K Accuracy')
                plt.legend()
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"Could not plot training history: {e}")


def main():
    # Hyperparameters optimized for 114k tokens
    predictor = NextWordPredictor(
        sequence_length=15,     # Short sequences for small dataset
        lstm_units=128,         # Smaller model to prevent overfitting
        vocab_size=1500         # Reasonable vocab for 114k tokens
    )

    # Download & preprocess
    raw_text = predictor.download_text()
    cleaned  = predictor.clean_text(raw_text)
    tokens   = predictor.tokenize_text(cleaned)

    # Load GloVe embeddings
    predictor.load_glove_embeddings("glove.6B.100d.txt", embedding_dim=100)

    # Prepare data - use more for training since dataset is small
    X, y = predictor.create_sequences(tokens)
    split = int(0.9 * len(X))  # Use 90% for training
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Dataset info: {len(tokens)} tokens, {len(X)} sequences")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Build & train
    predictor.model = predictor.build_mto_model()
    history = predictor.train_model(
        X_train, y_train,
        epochs=40,  # Moderate epochs for small dataset
        validation_split=0.2
    )

    # Load best model
    predictor.model.load_weights('best_model.h5')

    eval_results = predictor.model.evaluate(X_test, y_test, verbose=0)
    test_loss = eval_results[0]
    test_acc = eval_results[1]
    test_top_k = eval_results[2] if len(eval_results) > 2 else None
    
    train_results = predictor.model.evaluate(X_train, y_train, verbose=0)
    train_loss = train_results[0]
    train_acc = train_results[1]
    train_top_k = train_results[2] if len(train_results) > 2 else None
    
    perp = predictor.calculate_perplexity(X_test, y_test)

    print("\nFinal Metrics:")
    print(f"Training Accuracy: {train_acc:.4f}")
    if train_top_k is not None:
        print(f"Training Top-K:    {train_top_k:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")
    if test_top_k is not None:
        print(f"Test Top-K:        {test_top_k:.4f}")
    print(f"Perplexity:        {perp:.2f}")


    # Generate examples
    seeds = ["I saw Holmes", "The detective entered", "My dear Watson"]
    for seed in seeds:
        gen, _ = predictor.generate_text(seed, max_phrase_len=30, temperature=0.8)
        print(f"\nSeed: \"{seed}\"")
        print(f"Generated ({len(gen.split())} words): {gen}")

    # Step-by-step for first seed
    print("\nStep-by-step top-5:")
    _, breakdown = predictor.generate_text(seeds[0], max_phrase_len=10, temperature=0.8)
    for i, top5 in enumerate(breakdown, 1):
        line = ", ".join(f"{w}:{p:.3f}" for w, p in top5)
        print(f"  Step {i:2d}: {line}")

    # Save & plot
    predictor.save_model()
    predictor.plot_training_history(history)


if __name__ == "__main__":
    main()