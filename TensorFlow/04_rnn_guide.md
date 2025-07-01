# Recurrent Neural Networks (RNNs) with TensorFlow

A comprehensive guide to building and training recurrent neural networks for sequential data processing using TensorFlow and Keras.

## Table of Contents

1. [Introduction to RNNs](#introduction-to-rnns)
2. [RNN Architecture Components](#rnn-architecture-components)
3. [Building RNNs with Keras](#building-rnns-with-keras)
4. [Sequential Data Preprocessing](#sequential-data-preprocessing)
5. [Training RNNs](#training-rnns)
6. [LSTM and GRU Networks](#lstm-and-gru-networks)
7. [Bidirectional RNNs](#bidirectional-rnns)
8. [Attention Mechanisms](#attention-mechanisms)
9. [Time Series Forecasting](#time-series-forecasting)
10. [Text Processing with RNNs](#text-processing-with-rnns)

## Introduction to RNNs

Recurrent Neural Networks are designed to process sequential data by maintaining internal memory through recurrent connections. They are particularly effective for time series, text, and speech data.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Verify GPU availability
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

## RNN Architecture Components

### Basic RNN Layer

```python
# Simple RNN layer
rnn_layer = layers.SimpleRNN(
    units=64,                    # Number of output units
    activation='tanh',           # Activation function
    return_sequences=False,      # Return full sequence or last output
    return_state=False,          # Return internal states
    input_shape=(None, 10)       # (timesteps, features)
)

# RNN layer with sequences
rnn_sequences = layers.SimpleRNN(
    units=32,
    return_sequences=True,       # Return output for each timestep
    input_shape=(None, 10)
)

# RNN layer with states
rnn_with_states = layers.SimpleRNN(
    units=16,
    return_sequences=True,
    return_state=True,           # Return both output and states
    input_shape=(None, 10)
)
```

### LSTM Layer

Long Short-Term Memory networks address the vanishing gradient problem in RNNs.

```python
# Basic LSTM layer
lstm_layer = layers.LSTM(
    units=64,
    activation='tanh',
    recurrent_activation='sigmoid',
    return_sequences=False,
    return_state=False,
    input_shape=(None, 10)
)

# LSTM with sequences
lstm_sequences = layers.LSTM(
    units=32,
    return_sequences=True,
    dropout=0.2,                 # Dropout on inputs
    recurrent_dropout=0.2,       # Dropout on recurrent connections
    input_shape=(None, 10)
)

# LSTM with states
lstm_with_states = layers.LSTM(
    units=16,
    return_sequences=True,
    return_state=True,
    input_shape=(None, 10)
)
```

### GRU Layer

Gated Recurrent Units are a simplified version of LSTM with fewer parameters.

```python
# Basic GRU layer
gru_layer = layers.GRU(
    units=64,
    activation='tanh',
    recurrent_activation='sigmoid',
    return_sequences=False,
    return_state=False,
    input_shape=(None, 10)
)

# GRU with sequences
gru_sequences = layers.GRU(
    units=32,
    return_sequences=True,
    dropout=0.2,
    recurrent_dropout=0.2,
    input_shape=(None, 10)
)

# GRU with states
gru_with_states = layers.GRU(
    units=16,
    return_sequences=True,
    return_state=True,
    input_shape=(None, 10)
)
```

## Building RNNs with Keras

### Simple RNN Architecture

```python
# Simple RNN for sequence classification
model = keras.Sequential([
    layers.Input(shape=(None, 10)),  # Variable length sequences
    layers.SimpleRNN(64, return_sequences=True),
    layers.SimpleRNN(32),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()
```

### LSTM Architecture

```python
# LSTM for sequence classification
lstm_model = keras.Sequential([
    layers.Input(shape=(None, 10)),
    layers.LSTM(128, return_sequences=True, dropout=0.2),
    layers.LSTM(64, return_sequences=True, dropout=0.2),
    layers.LSTM(32, dropout=0.2),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

lstm_model.summary()
```

### GRU Architecture

```python
# GRU for sequence classification
gru_model = keras.Sequential([
    layers.Input(shape=(None, 10)),
    layers.GRU(128, return_sequences=True, dropout=0.2),
    layers.GRU(64, return_sequences=True, dropout=0.2),
    layers.GRU(32, dropout=0.2),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

gru_model.summary()
```

### Advanced RNN Architecture

```python
# Advanced RNN with multiple layers and regularization
def create_advanced_rnn(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First LSTM layer
        layers.LSTM(256, return_sequences=True, dropout=0.3),
        layers.BatchNormalization(),
        
        # Second LSTM layer
        layers.LSTM(128, return_sequences=True, dropout=0.3),
        layers.BatchNormalization(),
        
        # Third LSTM layer
        layers.LSTM(64, return_sequences=True, dropout=0.3),
        layers.BatchNormalization(),
        
        # Fourth LSTM layer
        layers.LSTM(32, dropout=0.3),
        layers.BatchNormalization(),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create advanced RNN model
advanced_rnn = create_advanced_rnn((None, 10), 5)
advanced_rnn.summary()
```

## Sequential Data Preprocessing

### Time Series Data

```python
# Generate synthetic time series data
def generate_time_series_data(n_samples, n_timesteps, n_features):
    np.random.seed(42)
    
    # Generate random sequences
    X = np.random.randn(n_samples, n_timesteps, n_features)
    
    # Create target: sum of features at each timestep
    y = np.sum(X, axis=(1, 2))
    
    # Convert to binary classification (above/below mean)
    y = (y > np.mean(y)).astype(int)
    
    return X, y

# Generate data
X_ts, y_ts = generate_time_series_data(1000, 50, 10)
print(f"Time series data shape: {X_ts.shape}, {y_ts.shape}")

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_ts, y_ts, test_size=0.2, random_state=42
)
```

### Text Data Preprocessing

```python
# Text preprocessing utilities
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data
texts = [
    "I love this movie, it's amazing!",
    "This is the worst film I've ever seen.",
    "Great acting and wonderful story.",
    "Terrible plot and bad acting.",
    "Excellent cinematography and direction.",
    "Boring and predictable storyline."
]

# Tokenize text
tokenizer = Tokenizer(num_words=1000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(texts)
print(f"Sequences: {sequences}")

# Pad sequences
padded_sequences = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')
print(f"Padded sequences shape: {padded_sequences.shape}")

# Create labels (positive/negative sentiment)
labels = np.array([1, 0, 1, 0, 1, 0])  # 1: positive, 0: negative
```

### Custom Data Generator

```python
# Custom data generator for sequential data
class SequentialDataGenerator:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
        self.indexes = np.arange(self.n_samples)
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.n_samples)
        
        batch_indexes = self.indexes[start_idx:end_idx]
        
        X_batch = self.X[batch_indexes]
        y_batch = self.y[batch_indexes]
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Create data generator
train_generator = SequentialDataGenerator(X_train, y_train, batch_size=32)
```

## Training RNNs

### Basic Training

```python
# Compile and train simple RNN
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    verbose=1
)
```

### Advanced Training with Callbacks

```python
# Define callbacks for RNN training
callbacks = [
    # Early stopping
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    
    # Learning rate scheduling
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    ),
    
    # Model checkpointing
    keras.callbacks.ModelCheckpoint(
        'best_rnn_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    
    # TensorBoard logging
    keras.callbacks.TensorBoard(log_dir='./logs/rnn')
]

# Train advanced RNN model
advanced_rnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = advanced_rnn.fit(
    X_train, y_train,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
```

## LSTM and GRU Networks

### LSTM for Sequence Classification

```python
# LSTM model for sentiment analysis
def create_lstm_sentiment_model(vocab_size, max_length, embedding_dim=100):
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.LSTM(128, return_sequences=True, dropout=0.2),
        layers.LSTM(64, dropout=0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# Create LSTM sentiment model
vocab_size = 1000
max_length = 10
lstm_sentiment = create_lstm_sentiment_model(vocab_size, max_length)
lstm_sentiment.summary()

# Compile and train
lstm_sentiment.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train on text data
lstm_sentiment.fit(
    padded_sequences, labels,
    epochs=20,
    validation_split=0.2,
    verbose=1
)
```

### GRU for Time Series Prediction

```python
# GRU model for time series prediction
def create_gru_timeseries_model(input_shape, output_steps):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(128, return_sequences=True, dropout=0.2),
        layers.GRU(64, return_sequences=True, dropout=0.2),
        layers.GRU(32, return_sequences=True, dropout=0.2),
        layers.Dense(output_steps)
    ])
    
    return model

# Create GRU time series model
input_shape = (50, 10)  # 50 timesteps, 10 features
output_steps = 5        # Predict next 5 steps
gru_timeseries = create_gru_timeseries_model(input_shape, output_steps)
gru_timeseries.summary()
```

## Bidirectional RNNs

Bidirectional RNNs process sequences in both forward and backward directions.

```python
# Bidirectional LSTM
bidirectional_lstm = keras.Sequential([
    layers.Input(shape=(None, 10)),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

bidirectional_lstm.summary()

# Bidirectional GRU
bidirectional_gru = keras.Sequential([
    layers.Input(shape=(None, 10)),
    layers.Bidirectional(layers.GRU(64, return_sequences=True)),
    layers.Bidirectional(layers.GRU(32)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

bidirectional_gru.summary()
```

## Attention Mechanisms

### Simple Attention Layer

```python
# Custom attention layer
class AttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, timesteps, features)
        e = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = tf.reduce_sum(inputs * a, axis=1)
        return output, a
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])

# RNN with attention
def create_rnn_with_attention(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # LSTM layer
    lstm_out = layers.LSTM(64, return_sequences=True)(inputs)
    
    # Attention layer
    attention_out, attention_weights = AttentionLayer()(lstm_out)
    
    # Dense layers
    dense_out = layers.Dense(32, activation='relu')(attention_out)
    outputs = layers.Dense(num_classes, activation='softmax')(dense_out)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create RNN with attention
rnn_attention = create_rnn_with_attention((None, 10), 5)
rnn_attention.summary()
```

## Time Series Forecasting

### Multi-step Time Series Prediction

```python
# Generate time series data for forecasting
def generate_forecasting_data(n_samples, n_timesteps, n_features, forecast_steps):
    np.random.seed(42)
    
    # Generate random sequences
    X = np.random.randn(n_samples, n_timesteps, n_features)
    
    # Create targets: next forecast_steps values
    y = np.random.randn(n_samples, forecast_steps)
    
    return X, y

# Generate forecasting data
X_forecast, y_forecast = generate_forecasting_data(1000, 50, 5, 10)
print(f"Forecasting data shape: {X_forecast.shape}, {y_forecast.shape}")

# LSTM for time series forecasting
forecasting_model = keras.Sequential([
    layers.Input(shape=(50, 5)),
    layers.LSTM(128, return_sequences=True, dropout=0.2),
    layers.LSTM(64, return_sequences=True, dropout=0.2),
    layers.LSTM(32, dropout=0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(10)  # Predict next 10 steps
])

forecasting_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Train forecasting model
forecasting_model.fit(
    X_forecast, y_forecast,
    batch_size=32,
    epochs=20,
    validation_split=0.2,
    verbose=1
)
```

### Sequence-to-Sequence Forecasting

```python
# Encoder-Decoder architecture for sequence-to-sequence
def create_seq2seq_model(input_shape, output_steps, latent_dim=64):
    # Encoder
    encoder_inputs = layers.Input(shape=input_shape)
    encoder_lstm = layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = layers.Input(shape=(output_steps, 1))
    decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = layers.Dense(1)
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Create model
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

# Create sequence-to-sequence model
seq2seq_model = create_seq2seq_model((50, 5), 10)
seq2seq_model.summary()
```

## Text Processing with RNNs

### Text Generation

```python
# Character-level text generation
def create_text_generation_model(vocab_size, sequence_length, embedding_dim=256):
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=sequence_length),
        layers.LSTM(256, return_sequences=True, dropout=0.2),
        layers.LSTM(256, return_sequences=True, dropout=0.2),
        layers.LSTM(256, dropout=0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    return model

# Text preprocessing for generation
def preprocess_text_for_generation(text, sequence_length=50):
    # Create character-level vocabulary
    chars = sorted(list(set(text)))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    # Convert text to sequences
    sequences = []
    next_chars = []
    
    for i in range(0, len(text) - sequence_length):
        sequences.append(text[i:i + sequence_length])
        next_chars.append(text[i + sequence_length])
    
    # Convert to numerical format
    X = np.zeros((len(sequences), sequence_length), dtype=int)
    y = np.zeros((len(sequences), len(chars)), dtype=int)
    
    for i, sequence in enumerate(sequences):
        for t, char in enumerate(sequence):
            X[i, t] = char_to_idx[char]
        y[i, char_to_idx[next_chars[i]]] = 1
    
    return X, y, char_to_idx, idx_to_char

# Sample text for generation
sample_text = """
This is a sample text for character-level text generation.
We will use this text to train a recurrent neural network.
The model will learn to predict the next character given a sequence of characters.
"""

# Preprocess text
X_text, y_text, char_to_idx, idx_to_char = preprocess_text_for_generation(sample_text)
vocab_size = len(char_to_idx)

# Create text generation model
text_gen_model = create_text_generation_model(vocab_size, 50)
text_gen_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train text generation model
text_gen_model.fit(
    X_text, y_text,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    verbose=1
)
```

### Text Classification

```python
# LSTM for text classification
def create_text_classification_model(vocab_size, max_length, num_classes, embedding_dim=100):
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.LSTM(128, return_sequences=True, dropout=0.2),
        layers.LSTM(64, dropout=0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create text classification model
vocab_size = 1000
max_length = 100
num_classes = 3

text_class_model = create_text_classification_model(vocab_size, max_length, num_classes)
text_class_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

text_class_model.summary()
```

## Summary

- RNNs are designed for processing sequential data with internal memory
- LSTM and GRU networks address the vanishing gradient problem
- Bidirectional RNNs process sequences in both directions
- Attention mechanisms help focus on relevant parts of sequences
- RNNs are effective for time series forecasting and text processing
- Proper data preprocessing is crucial for RNN performance

## Next Steps

- Explore transformer architectures
- Learn about sequence-to-sequence models
- Practice with real-world time series datasets
- Implement advanced attention mechanisms 