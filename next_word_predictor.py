#!/usr/bin/env python3
# next_word_predictor.py - Improved version with attention mechanism and advanced text generation

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Embedding, Dropout,
    Input, Add, LayerNormalization,
    MultiHeadAttention, Layer
)
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import requests 
import re
from collections import Counter, deque
import pickle
import matplotlib.pyplot as plt
import logging

BATCH_SIZE = 32

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


class PositionalEncoding(Layer):
    #Custom Keras layer to add positional embedding layer for context awareness
    def __init__(self, max_seq_len, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        
        # Create positional encoding matrix
        pos_encoding = np.zeros((max_seq_len, embedding_dim))
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** (i / embedding_dim)))
                if i + 1 < embedding_dim:
                    pos_encoding[pos, i + 1] = np.cos(pos / (10000 ** (i / embedding_dim)))
        
        self.pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
    
    def call(self, inputs):
        # Add positional encoding to inputs at each time step based on sequence length
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:seq_len, :]


class NextWordPredictor:
    def __init__(self, sequence_length=15, lstm_units=128, vocab_size=1500):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.vocab_size = vocab_size
        self.model = None
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embedding_matrix = None
        self.embedding_dim = None

    # Download text from Project Gutenberg
    def download_text(self, url="https://www.gutenberg.org/files/1661/1661-0.txt"):
        logger.info("Downloading text from Project Gutenberg...")
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def clean_text(self, text):
        logger.info("Cleaning text...")

        #Strip headers and footers from Project Gutenberg text
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker   = "*** END OF THE PROJECT GUTENBERG EBOOK"
        start_idx = text.find(start_marker)
        end_idx   = text.find(end_marker)
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx + len(start_marker) : end_idx]
        
        # Clean but preserve sentence structure
        text = re.sub(r'\r\n', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"\(\)]', '', text)
        text = re.sub(r'_{2,}', '', text)
        text = re.sub(r'\*{2,}', '', text)
        text = re.sub(r'CHAPTER\s+[IVX]+', '', text)
        text = text.lower().strip()
        
        sentences = re.split(r'[.!?]+', text)
        return '. '.join(sentences)

    def tokenize_text(self, text):
        logger.info("Tokenizing text...")
        words = re.findall(r'\b\w+\b|[.!?,:;]', text)
        words = [w for w in words if len(w) > 0]
        
        counts = Counter(words)
        
        # Keep special 1-character words like "i" and "a"
        filtered_counts = {}
        for word, count in counts.items():
            if (word.lower() in ['i', 'a'] and count >= 2) or (count >= 3 and len(word) >= 1):
                filtered_counts[word] = count
        
        # Build vocabulary
        most_common = Counter(filtered_counts).most_common(self.vocab_size - 1)
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        for idx, (w, _) in enumerate(most_common):
            self.word_to_idx[w] = idx
            self.idx_to_word[idx] = w
        
        # Convert to indices
        tokenized = []
        for w in words:
            if w in self.word_to_idx:
                tokenized.append(self.word_to_idx[w])
        
        coverage = len(tokenized) / len(words) * 100
        logger.info(f"Vocab size: {len(self.word_to_idx)}, tokens: {len(tokenized)}, coverage: {coverage:.2f}%")
        
        return tokenized

    def create_sequences(self, tokenized_text):
        logger.info("Creating many-to-many sequences...")
        seqs, targets = [], []
        
        #Shorter stride to capture more context
        stride = max(1, self.sequence_length // 4)
        
        #Generate sequences of size sequence_length by iterating through the tokenized text
        for i in range(0, len(tokenized_text) - self.sequence_length, stride):
            seq = tokenized_text[i:i + self.sequence_length]
            target = tokenized_text[i + 1:i + self.sequence_length + 1]
            
            seqs.append(seq)
            targets.append(target)
        
        X = np.array(seqs)
        y = np.array(targets)

        #Convert to one-hot vectors
        y = tf.keras.utils.to_categorical(y, num_classes=len(self.word_to_idx))
        
        logger.info(f"  → {len(seqs)} samples, input shape: {X.shape}, target shape: {y.shape}")
        return X, y

    def load_glove_embeddings(self, glove_path: str, embedding_dim=100):
        glove_index = load_glove_file(glove_path)
        self.embedding_dim = embedding_dim
        vocab_size = len(self.word_to_idx)
        
        # Initialize with small random values
        self.embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim)).astype('float32')
        
        hits = 0
        for word, idx in self.word_to_idx.items():
            vec = glove_index.get(word)
            if vec is not None:
                self.embedding_matrix[idx] = vec
                hits += 1
        
        logger.info(f"Built embedding matrix with shape {self.embedding_matrix.shape}")
        logger.info(f"GloVe hits: {hits}/{vocab_size} ({hits/vocab_size*100:.1f}%)")

    def build_model(self, hidden_size=None, optimizer=None):
        logger.info("Building model...")

        # Set default values
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
            mask_zero=False
        )(inputs)

        # Add positional encoding
        x = PositionalEncoding(self.sequence_length, self.embedding_dim)(x)
        x = Dropout(0.2)(x)

        # LSTM layers
        lstm_out = LSTM(hidden_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
        lstm_out = LSTM(hidden_size, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(lstm_out)
        
        # Add attention mechanism -- multi-head self attention
        attention_out = MultiHeadAttention(
                num_heads=4, 
                key_dim=hidden_size // 4,
                dropout=0.1
        )(lstm_out, lstm_out)
        
        # Add residual connection and layer normalization
        attention_out = Add()([lstm_out, attention_out])
        attention_out = LayerNormalization()(attention_out)

        # Add dropout
        x = Dropout(0.3)(attention_out)

        # Output layer
        outputs = Dense(len(self.word_to_idx), activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        self.model = model
        logger.info(f"Built improved many-to-many LSTM model with self-attention mechanism")
        print(f"→ Model params: {model.count_params():,}")
        return model

    def train_model(self, X, y, epochs, validation_split=0.2):
        logger.info("Training model...")
        monitor_val = 'val_accuracy'
        
        #Early stopping to avoid overfitting
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
        
        #Checkpoint to save best model
        ckpt = ModelCheckpoint(
            'best_model.h5', 
            monitor=monitor_val,
            save_best_only=True, 
            mode='max',
            verbose=1
        )
        
        #Reduce learning rate if loss does not improve
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.7,
            patience=4,
            min_lr=1e-6, 
            verbose=1
        )
        
        #Fit model
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
        ce = 0.0    #Cross-entropy
        n = 0       #Number of words
        
        #Iterate through the dataset in batches
        for i in range(0, len(X), BATCH_SIZE):
            bx = X[i:i+BATCH_SIZE]
            by = y[i:i+BATCH_SIZE]
            preds = self.model.predict(bx, verbose=0)
            
            #For each true word index in the batch, calculate the cross-entropy
            for j in range(len(by)):
                for k in range(by.shape[1]):
                    true_idx = np.argmax(by[j, k])
                    #Get the predicted probability of the true word
                    p = preds[j, k, true_idx]
                    #Add the negative log of the predicted probability to the total cross-entropy
                    ce += -np.log(max(p, 1e-10))
                    #Increment the counter for the total number of words
                    n += 1
        
        #Calculate the perplexity by taking the average of the cross-entropy
        ce /= n
        perp = np.exp(ce)
        logger.info(f"Perplexity: {perp:.2f}")
        return perp

    #Perform nucleus sampling to select an index based on the cumulative probability distribution.
    def nucleus_sampling(self, probs, p=0.9):
        #Sort the probabilities in descending order
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        
        #Calculate the cumulative probability distribution
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum, p) + 1
        cutoff_idx = max(cutoff_idx, 1)
        
        #Select an index based on the cutoff
        top_p_indices = sorted_indices[:cutoff_idx]
        top_p_probs = sorted_probs[:cutoff_idx]
        top_p_probs = top_p_probs / np.sum(top_p_probs)
        
        #Choose a random index from the top-p indices
        chosen_idx = np.random.choice(len(top_p_indices), p=top_p_probs)

        return top_p_indices[chosen_idx]

    def apply_repetition_penalty(self, probs, recent_words, all_generated_words, last_word_idx=None, last_two_words=None, sentence_position=0):
        
        # Strong penalty for recent words
        for word_idx in recent_words:
            if word_idx < len(probs):
                probs[word_idx] *= 0.02

        # Distance-based repetition penalty
        word_counts = Counter(all_generated_words)
        for word_idx, count in word_counts.items():
            if word_idx < len(probs) and count >= 1:
                distance_factor = min(10, len(all_generated_words) - max([i for i, w in enumerate(all_generated_words) if w == word_idx]))
                penalty = (0.15 ** count) * (0.7 ** (10 - distance_factor))
                probs[word_idx] *= penalty

        # Eliminate immediate repetition
        if last_word_idx is not None and last_word_idx < len(probs):
            probs[last_word_idx] *= 0.0001

        # Grammar-aware boosting
        if last_two_words is not None and len(last_two_words) >= 1:
            last_word = self.idx_to_word.get(last_two_words[-1], '')
            
            grammar_boosts = {
                'not': ['be', 'have', 'to', 'a', 'the', 'in', 'at', 'of'],
                'to': ['be', 'have', 'see', 'go', 'come', 'take', 'make', 'find', 'do', 'get'],
                'had': ['been', 'done', 'seen', 'found', 'taken', 'made', 'come', 'gone', 'heard', 'not'],
                'have': ['been', 'done', 'seen', 'found', 'taken', 'made', 'come', 'gone', 'heard', 'not'],
                'was': ['a', 'the', 'not', 'very', 'quite', 'rather', 'indeed', 'in', 'at', 'no'],
                'is': ['a', 'the', 'not', 'very', 'quite', 'rather', 'indeed', 'in', 'at', 'no'],
                'i': ['am', 'was', 'have', 'had', 'shall', 'will', 'think', 'know', 'see', 'found', 'heard'],
                'he': ['was', 'had', 'came', 'went', 'looked', 'said', 'took', 'made', 'found', 'could'],
                'it': ['was', 'is', 'had', 'seemed', 'appeared', 'would', 'could', 'might'],
                'the': ['man', 'woman', 'case', 'matter', 'door', 'room', 'house', 'mystery', 'evidence', 'time'],
                'there': ['was', 'were', 'had', 'came', 'appeared', 'seemed', 'is', 'are'],
                'and': ['i', 'he', 'she', 'it', 'the', 'in', 'then', 'yet', 'so', 'now'],
                'very': ['much', 'well', 'good', 'little', 'few', 'many', 'strange', 'curious', 'peculiar'],
                'quite': ['right', 'sure', 'certain', 'clear', 'true', 'well', 'good', 'so'],
            }
            
            if last_word in grammar_boosts:
                for good_word in grammar_boosts[last_word]:
                    if good_word in self.word_to_idx:
                        good_idx = self.word_to_idx[good_word]
                        if good_idx < len(probs):
                            probs[good_idx] *= 2.0

        # Eliminate problematic patterns
        if last_two_words is not None and len(last_two_words) >= 2:
            forbidden_patterns = {
                ('not', 'be'): ['to', 'in', 'a', 'the', 'no', 'me', 'you'],
                ('be', 'to'): ['the', 'a', 'in', 'no', 'me', 'you'],
                ('to', 'be'): ['to', 'be', 'in', 'no', 'me', 'you', 'not'],
                ('am', 'not'): ['to', 'in', 'a', 'no', 'me', 'you'],
                ('been', 'not'): ['to', 'in', 'a', 'no', 'me', 'you'],
                ('have', 'been'): ['not', 'to', 'in', 'a', 'the', 'no'],
                ('had', 'been'): ['not', 'to', 'in', 'a', 'the', 'no'],
                ('was', 'not'): ['to', 'in', 'a', 'no', 'me', 'you'],
                ('is', 'not'): ['to', 'in', 'a', 'no', 'me', 'you'],
                ('you', 'have'): ['not', 'been', 'to', 'in', 'a'],
                ('that', 'he'): ['was', 'had', 'is', 'have'],
                ('that', 'it'): ['was', 'had', 'is', 'have'],
                ('that', 'you'): ['have', 'are', 'were', 'had'],
                ('and', 'i'): ['am', 'have', 'had', 'was'],
                ('and', 'it'): ['was', 'is', 'had', 'have'],
                ('of', 'a'): ['man', 'case', 'very', 'good', 'great'],
                ('in', 'the'): ['case', 'matter', 'house', 'room', 'time'],
            }
            
            last_word = self.idx_to_word.get(last_two_words[-1], '')
            second_last = self.idx_to_word.get(last_two_words[-2], '') if len(last_two_words) >= 2 else ''
            
            pattern = (second_last, last_word)
            if pattern in forbidden_patterns:
                for forbidden_word in forbidden_patterns[pattern]:
                    if forbidden_word in self.word_to_idx:
                        forbidden_idx = self.word_to_idx[forbidden_word]
                        if forbidden_idx < len(probs):
                            probs[forbidden_idx] *= 0.0001

        # Handle punctuation
        punctuation_indices = []
        for word, idx in self.word_to_idx.items():
            if word in [',', '.', '!', '?', ';', ':'] and idx < len(probs):
                punctuation_indices.append(idx)

        if last_word_idx is not None and last_word_idx in punctuation_indices:
            for punct_idx in punctuation_indices:
                probs[punct_idx] *= 0.0001

        # Boost quality vocabulary
        quality_words = {
            'action_verbs': ['examined', 'discovered', 'revealed', 'noticed', 'observed', 'concluded',
                           'deduced', 'investigated', 'approached', 'entered', 'departed', 'returned',
                           'exclaimed', 'declared', 'announced', 'whispered', 'murmured', 'replied'],
            'descriptive': ['mysterious', 'strange', 'peculiar', 'remarkable', 'extraordinary', 'curious',
                          'unusual', 'suspicious', 'evident', 'obvious', 'clear', 'distinct', 'singular',
                          'interesting', 'impressive', 'striking'],
            'story_elements': ['clue', 'evidence', 'mystery', 'solution', 'problem', 'question', 'answer',
                             'truth', 'fact', 'detail', 'circumstance', 'situation', 'incident', 'affair',
                             'business', 'adventure', 'investigation', 'inquiry'],
            'character_words': ['gentleman', 'fellow', 'companion', 'client', 'visitor', 'stranger',
                              'inspector', 'detective', 'doctor', 'professor', 'lady', 'woman'],
        }

        boost_factor = 3.0 if sentence_position <= 3 else 2.0

        for _, words in quality_words.items():
            for word in words:
                if word in self.word_to_idx:
                    word_idx = self.word_to_idx[word]
                    if word_idx < len(probs) and word_idx not in all_generated_words:
                        probs[word_idx] *= boost_factor

        return probs

    def post_process_text(self, text):
        # Capitalize 'i' when it appears as a standalone word
        text = re.sub(r'\bi\b', 'I', text)
        
        # Capitalize first letter after sentence-ending punctuation
        text = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
        
        # Capitalize the very first letter if it's lowercase
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Fix spacing issues around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Fix double spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def generate_text(self, seed_text, number_of_words=None, max_phrase_len=50, temperature=0.4,nucleus_p=0.75, post_process=True, analyze=False):
        model = self.model
        seed_words = re.findall(r'\b\w+\b|[.!?,:;]', seed_text.lower())
        
        # Only use words in vocabulary
        seed_tokens = []
        for w in seed_words:
            if w in self.word_to_idx:
                seed_tokens.append(self.word_to_idx[w])
        
        if not seed_tokens:
            seed_tokens = [self.word_to_idx.get('the', 0)]
        
        # Pad sequence if too short
        if len(seed_tokens) < self.sequence_length:
            pad_token = 0
            seed_tokens = [pad_token] * (self.sequence_length - len(seed_tokens)) + seed_tokens

        generated = seed_text
        analysis_steps = [] if analyze else None
        recent_words = deque(maxlen=8)
        all_generated_words = []
        words_generated = 0
        last_word_idx = None
        last_two_words = deque(maxlen=2)
        sentence_position = 0
        
        target_words = number_of_words if number_of_words is not None else max_phrase_len
        max_iterations = target_words * 6
        iteration = 0
        consecutive_punctuation = 0
        failed_attempts = 0
        
        while words_generated < target_words and iteration < max_iterations and failed_attempts < 40:
            iteration += 1
            
            # Get model predictions
            current_sequence = seed_tokens[-self.sequence_length:]
            inp = np.array([current_sequence])
            preds = model.predict(inp, verbose=0)
            
            # Store original predictions for analysis
            original_preds = preds[0, -1, :].copy() if analyze else None
            preds = preds[0, -1, :]
            
            # Apply penalties and boosts
            preds = self.apply_repetition_penalty(
                preds, recent_words, all_generated_words, last_word_idx, last_two_words, sentence_position
            )
            
            # Apply dynamic temperature
            effective_temp = temperature * (1.0 + 0.1 * min(sentence_position, 5))
            
            if effective_temp > 0:
                preds = np.log(preds + 1e-10) / effective_temp
                preds = np.exp(preds) / np.sum(np.exp(preds))

            # Prepare analysis data if requested
            if analyze:
                top_5_idx = np.argsort(preds)[-5:][::-1]
                top_5_original = np.argsort(original_preds)[-5:][::-1]
                
                step_analysis = {
                    'step': words_generated + 1,
                    'context': ' '.join([self.idx_to_word.get(idx, '<UNK>') for idx in current_sequence[-5:]]),
                    'top_5_original': [
                        (self.idx_to_word.get(idx, '<UNK>'), float(original_preds[idx]))
                        for idx in top_5_original
                    ],
                    'top_5_after_penalties': [
                        (self.idx_to_word.get(idx, '<UNK>'), float(preds[idx]))
                        for idx in top_5_idx
                    ],
                    'recent_words': [self.idx_to_word.get(idx, '<UNK>') for idx in list(recent_words)],
                    'failed_attempts': failed_attempts
                }

            # Sample next word
            next_idx = self.nucleus_sampling(preds, p=nucleus_p)
            next_word = self.idx_to_word.get(next_idx, '')
            
            if not next_word:
                failed_attempts += 1
                if analyze:
                    step_analysis['action'] = 'SKIP: Word not found'
                    analysis_steps.append(step_analysis)
                continue

            # Quality checks
            is_punctuation = next_word in ['.', '!', '?', ',', ';', ':']
            is_sentence_ender = next_word in ['.', '!', '?']
            
            skip_word = False
            skip_reason = ""
            
            if len(last_two_words) >= 1:
                last_word = self.idx_to_word.get(last_two_words[-1], '')
                
                # Check for bad patterns
                bad_combinations = [
                    ('of', 'of'), ('the', 'the'), ('and', 'and'), ('in', 'in'), ('to', 'to'),
                    ('i', 'i'), ('a', 'a'), ('is', 'is'), ('was', 'was'), ('a', 'of'),
                    ('be', 'to'), ('had', 'be'), ('have', 'be'), ('is', 'to'), ('was', 'to'),
                    ('and', 'and'), ('said', 'said'), ('have', 'have'), ('had', 'had')
                ]
                
                if (last_word, next_word) in bad_combinations:
                    skip_word = True
                    skip_reason = f"Bad pattern: '{last_word} {next_word}'"
                
                # Check for overused pairs
                if (last_word, next_word) in [('said', 'he'), ('i', 'am'), ('it', 'is'), ('he', 'was')]:
                    pair_count = sum(1 for i in range(len(all_generated_words)-1) 
                                   if (self.idx_to_word.get(all_generated_words[i], ''), 
                                       self.idx_to_word.get(all_generated_words[i+1], '')) == (last_word, next_word))
                    if pair_count >= 2:
                        skip_word = True
                        skip_reason = f"Overused pair: '{last_word} {next_word}'"

            if is_punctuation and consecutive_punctuation >= 1:
                skip_word = True
                skip_reason = "Consecutive punctuation"

            # Encourage sentence endings for long sentences
            if sentence_position > 15 and not is_sentence_ender:
                if '.' in self.word_to_idx and iteration % 3 == 0:
                    period_idx = self.word_to_idx['.']
                    if period_idx < len(preds):
                        preds[period_idx] *= 5.0

            if skip_word and failed_attempts < 15:
                failed_attempts += 1
                if analyze:
                    step_analysis['action'] = f'SKIP: {skip_reason}'
                    step_analysis['chosen_word'] = next_word
                    analysis_steps.append(step_analysis)
                continue

            # Accept the word
            if analyze:
                step_analysis['chosen_word'] = next_word
                step_analysis['action'] = 'ACCEPT'
                analysis_steps.append(step_analysis)

            if is_punctuation:
                generated += next_word
                consecutive_punctuation += 1
                if is_sentence_ender:
                    sentence_position = 0
                else:
                    sentence_position += 1
            else:
                generated += " " + next_word
                words_generated += 1
                consecutive_punctuation = 0
                sentence_position += 1

            failed_attempts = 0

            # Update tracking
            recent_words.append(next_idx)
            all_generated_words.append(next_idx)
            last_word_idx = next_idx
            last_two_words.append(next_idx)
            seed_tokens.append(next_idx)

            if len(seed_tokens) > self.sequence_length * 2:
                seed_tokens = seed_tokens[-self.sequence_length:]

        # Apply post-processing
        if post_process:
            generated = self.post_process_text(generated)

        # Return results
        if analyze:
            return generated, analysis_steps
        else:
            return generated

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

def main():
    # Initialize predictor object
    predictor = NextWordPredictor(
        sequence_length=15,
        lstm_units=128,
        vocab_size=3000
    )

    # Download and preprocess data
    raw_text = predictor.download_text()
    cleaned = predictor.clean_text(raw_text)
    tokens = predictor.tokenize_text(cleaned)

    # Load GloVe embeddings
    predictor.load_glove_embeddings("glove.6B.100d.txt", embedding_dim=100)

    # Prepare training data
    X, y = predictor.create_sequences(tokens)
    split = int(0.9 * len(X)) #Since dataset is so small, need as much training data as possible
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Dataset info: {len(tokens)} tokens, {len(X)} sequences")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Build model with attention
    predictor.model = predictor.build_model()
    
    #If training, uncomment next lines (all of history object initialization)
    #history = predictor.train_model(
     #   X_train, y_train,
      #  epochs=30,
       # validation_split=0.2
    #)

    # Load best model (if not training - uncomment next line)
    predictor.model.load_weights('best_model.h5')

    # Evaluate model performance
    eval_results = predictor.model.evaluate(X_test, y_test, verbose=0)
    test_acc = eval_results[1]
    
    train_results = predictor.model.evaluate(X_train, y_train, verbose=0)
    train_acc = train_results[1]
    
    perp = predictor.calculate_perplexity(X_test, y_test)

    print("\nFinal Metrics:")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")
    print(f"Perplexity:        {perp:.2f}")

    #Generate longer examples (30+ words)
    print("\n" + "="*80)
    print("LONGER TEXT GENERATION EXAMPLES (30+ words)")
    print("="*80)
    
    longer_seeds = [
        "The mystery began when Holmes",
        "Watson examined the evidence",
        "In the dimly lit room",
        "The stranger approached cautiously",
        "My dear fellow, this case"
    ]
    
    for i, seed in enumerate(longer_seeds, 1):
        print(f"\n--- Example {i} ---")
        print(f"Seed: \"{seed}\"")
        
        generated_text = predictor.generate_text(
            seed, 
            number_of_words=32, 
            temperature=0.4, 
            nucleus_p=0.8, 
            post_process=True
        )
        
        word_count = len(re.findall(r'\b\w+\b', generated_text)) - len(re.findall(r'\b\w+\b', seed))
        print(f"Generated ({word_count} new words): {generated_text}")
        print()

    #Detailed step-by-step analysis
    print("\n" + "="*80)
    print("STEP-BY-STEP MODEL DECISION ANALYSIS")
    print("="*80)
    
    analysis_seed = "Holmes studied the peculiar"
    print(f"Analyzing generation for seed: \"{analysis_seed}\"")
    print(f"Target: 20 words\n")
    
    generated_text, analysis_steps = predictor.generate_text(
        analysis_seed, 
        number_of_words=20, 
        temperature=0.4, 
        nucleus_p=0.8,
        analyze=True  # Enable analysis mode
    )
    
    print(f"FINAL GENERATED TEXT: {generated_text}\n")
    print("DETAILED STEP-BY-STEP ANALYSIS:")
    print("-" * 80)
    
    accepted_steps = [step for step in analysis_steps if step['action'] == 'ACCEPT']
    
    for step in accepted_steps:
        print(f"\nStep {step['step']}:")
        print(f"Context (last 5 words): ...{step['context']}")
        print(f"Recent words tracked: {step['recent_words']}")
        
        print(f"\nTop 5 predictions BEFORE penalties:")
        for i, (word, prob) in enumerate(step['top_5_original'], 1):
            print(f"  {i}. '{word}' (prob: {prob:.4f})")
        
        print(f"\nTop 5 predictions AFTER penalties & temperature:")
        for i, (word, prob) in enumerate(step['top_5_after_penalties'], 1):
            print(f"  {i}. '{word}' (prob: {prob:.4f})")
        
        print(f"\nCHOSEN: '{step['chosen_word']}' - {step['action']}")
        
        if step['failed_attempts'] > 0:
            print(f"Failed attempts before this choice: {step['failed_attempts']}")
        
        print("-" * 40)

    # Show rejected examples
    skipped_steps = [step for step in analysis_steps if step['action'].startswith('SKIP')]
    if skipped_steps:
        print(f"\nEXAMPLES OF REJECTED WORDS (showing quality control):")
        for step in skipped_steps[:3]:
            print(f"Step {step.get('step', 'N/A')}: Rejected '{step['chosen_word']}' - {step['action']}")

    print(f"\nSUMMARY:")
    print(f"- Total analysis steps: {len(analysis_steps)}")
    print(f"- Accepted words: {len(accepted_steps)}")
    print(f"- Rejected words: {len(skipped_steps)}")
    print(f"- Success rate: {len(accepted_steps)/(len(analysis_steps))*100:.1f}%")

    # Save model
    predictor.save_model()


if __name__ == "__main__":
    main()