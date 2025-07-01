# Natural Language Processing Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Level](https://img.shields.io/badge/Level-Advanced-red.svg)](https://github.com/yourusername/Toolkit)
[![NLTK](https://img.shields.io/badge/NLTK-3.7%2B-green.svg)](https://www.nltk.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.4%2B-blue.svg)](https://spacy.io/)
[![Transformers](https://img.shields.io/badge/Transformers-4.20%2B-orange.svg)](https://huggingface.co/transformers/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![Topics](https://img.shields.io/badge/Topics-NLP%2C%20Text%20Processing%2C%20Language%20Models-orange.svg)](https://github.com/yourusername/Toolkit)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/yourusername/Toolkit)

A comprehensive guide to natural language processing techniques and applications using Python for AI/ML projects.

## Table of Contents
1. [Introduction to NLP](#introduction-to-nlp)
2. [Text Preprocessing](#text-preprocessing)
3. [Text Representation](#text-representation)
4. [Language Models](#language-models)
5. [Text Classification](#text-classification)
6. [Sentiment Analysis](#sentiment-analysis)
7. [Named Entity Recognition](#named-entity-recognition)
8. [Text Generation](#text-generation)
9. [Best Practices](#best-practices)

## Introduction to NLP

### What is Natural Language Processing?

Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, and manipulate human language.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Machine learning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Transformers
from transformers import pipeline, AutoTokenizer, AutoModel

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Download NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
except:
    print("NLTK data already downloaded or download failed")
```

### Basic Text Operations

```python
def basic_text_operations(text):
    """
    Perform basic text operations
    
    Args:
        text: Input text string
    
    Returns:
        Dictionary with basic text information
    """
    info = {
        'length': len(text),
        'word_count': len(text.split()),
        'sentence_count': len(sent_tokenize(text)),
        'character_count': len(text.replace(' ', '')),
        'unique_words': len(set(text.lower().split())),
        'average_word_length': np.mean([len(word) for word in text.split()])
    }
    
    return info

def analyze_text_statistics(text):
    """
    Analyze text statistics
    
    Args:
        text: Input text string
    
    Returns:
        Dictionary with text statistics
    """
    # Tokenize
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    # Word frequency
    word_freq = Counter(words)
    most_common = word_freq.most_common(10)
    
    # Character frequency
    char_freq = Counter(text.lower())
    
    # Statistics
    stats = {
        'total_words': len(words),
        'total_sentences': len(sentences),
        'average_sentence_length': len(words) / len(sentences),
        'most_common_words': most_common,
        'vocabulary_size': len(set(words)),
        'lexical_diversity': len(set(words)) / len(words)
    }
    
    return stats

# Example usage
sample_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and 
human language, in particular how to program computers to process and analyze 
large amounts of natural language data.
"""

text_info = basic_text_operations(sample_text)
text_stats = analyze_text_statistics(sample_text)

print("Text Information:")
for key, value in text_info.items():
    print(f"{key}: {value}")

print("\nText Statistics:")
for key, value in text_stats.items():
    print(f"{key}: {value}")
```

## Text Preprocessing

### Text Cleaning

```python
def clean_text(text):
    """
    Clean and normalize text
    
    Args:
        text: Input text string
    
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text

def remove_stopwords(text):
    """
    Remove stopwords from text
    
    Args:
        text: Input text string
    
    Returns:
        Text without stopwords
    """
    # Tokenize
    words = word_tokenize(text.lower())
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

def apply_stemming(text):
    """
    Apply stemming to text
    
    Args:
        text: Input text string
    
    Returns:
        Stemmed text
    """
    # Initialize stemmer
    stemmer = PorterStemmer()
    
    # Tokenize
    words = word_tokenize(text)
    
    # Apply stemming
    stemmed_words = [stemmer.stem(word) for word in words]
    
    return ' '.join(stemmed_words)

def apply_lemmatization(text):
    """
    Apply lemmatization to text
    
    Args:
        text: Input text string
    
    Returns:
        Lemmatized text
    """
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize
    words = word_tokenize(text)
    
    # Apply lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(lemmatized_words)

def comprehensive_text_preprocessing(text):
    """
    Apply comprehensive text preprocessing
    
    Args:
        text: Input text string
    
    Returns:
        Preprocessed text
    """
    # Clean text
    cleaned = clean_text(text)
    
    # Remove stopwords
    no_stopwords = remove_stopwords(cleaned)
    
    # Apply lemmatization
    lemmatized = apply_lemmatization(no_stopwords)
    
    return lemmatized

# Example usage
sample_texts = [
    "The quick brown fox jumps over the lazy dog!",
    "I am running in the park with my friends.",
    "Natural Language Processing is amazing!!!"
]

for text in sample_texts:
    print(f"Original: {text}")
    print(f"Cleaned: {comprehensive_text_preprocessing(text)}")
    print("-" * 50)
```

### Advanced Text Preprocessing

```python
def extract_pos_tags(text):
    """
    Extract part-of-speech tags
    
    Args:
        text: Input text string
    
    Returns:
        List of (word, tag) tuples
    """
    # Tokenize
    words = word_tokenize(text)
    
    # Get POS tags
    pos_tags = pos_tag(words)
    
    return pos_tags

def extract_named_entities(text):
    """
    Extract named entities
    
    Args:
        text: Input text string
    
    Returns:
        Named entities
    """
    # Tokenize and tag
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    
    # Extract named entities
    named_entities = ne_chunk(pos_tags)
    
    return named_entities

def create_preprocessing_pipeline():
    """
    Create a comprehensive text preprocessing pipeline
    
    Returns:
        Preprocessing functions
    """
    def pipeline(text):
        # Step 1: Basic cleaning
        text = clean_text(text)
        
        # Step 2: Remove stopwords
        text = remove_stopwords(text)
        
        # Step 3: Lemmatization
        text = apply_lemmatization(text)
        
        return text
    
    return pipeline

# Example usage
pipeline = create_preprocessing_pipeline()

sample_text = "The Natural Language Processing algorithms are running efficiently!"
processed_text = pipeline(sample_text)
print(f"Original: {sample_text}")
print(f"Processed: {processed_text}")

# POS tagging
pos_tags = extract_pos_tags(sample_text)
print(f"POS Tags: {pos_tags}")
```

## Text Representation

### Bag of Words

```python
def create_bag_of_words(texts):
    """
    Create bag of words representation
    
    Args:
        texts: List of text strings
    
    Returns:
        Bag of words matrix and vocabulary
    """
    # Initialize vectorizer
    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit and transform
    bow_matrix = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.get_feature_names_out()
    
    return bow_matrix, vocabulary

def create_tfidf_representation(texts):
    """
    Create TF-IDF representation
    
    Args:
        texts: List of text strings
    
    Returns:
        TF-IDF matrix and vocabulary
    """
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit and transform
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    vocabulary = tfidf_vectorizer.get_feature_names_out()
    
    return tfidf_matrix, vocabulary

def analyze_text_vectors(texts):
    """
    Analyze text vector representations
    
    Args:
        texts: List of text strings
    
    Returns:
        Analysis results
    """
    # Create different representations
    bow_matrix, bow_vocab = create_bag_of_words(texts)
    tfidf_matrix, tfidf_vocab = create_tfidf_representation(texts)
    
    # Analysis
    analysis = {
        'bow_shape': bow_matrix.shape,
        'tfidf_shape': tfidf_matrix.shape,
        'bow_vocabulary_size': len(bow_vocab),
        'tfidf_vocabulary_size': len(tfidf_vocab),
        'bow_density': bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1]),
        'tfidf_density': tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
    }
    
    return analysis, bow_matrix, tfidf_matrix

# Example usage
sample_texts = [
    "Natural language processing is a subfield of artificial intelligence",
    "Machine learning algorithms can process text data effectively",
    "Deep learning models have revolutionized NLP applications",
    "Text preprocessing is essential for NLP tasks"
]

analysis, bow_matrix, tfidf_matrix = analyze_text_vectors(sample_texts)

print("Text Vector Analysis:")
for key, value in analysis.items():
    print(f"{key}: {value}")
```

### Word Embeddings

```python
def create_word_embeddings(texts, max_words=1000, max_len=100):
    """
    Create word embeddings using Keras Tokenizer
    
    Args:
        texts: List of text strings
        max_words: Maximum number of words in vocabulary
        max_len: Maximum sequence length
    
    Returns:
        Tokenized sequences and tokenizer
    """
    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    
    # Fit on texts
    tokenizer.fit_on_texts(texts)
    
    # Convert to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return padded_sequences, tokenizer

def create_embedding_model(vocab_size, embedding_dim=100, max_len=100):
    """
    Create a simple embedding model
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension
        max_len: Maximum sequence length
    
    Returns:
        Embedding model
    """
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Example usage
sample_texts = [
    "I love this movie",
    "This movie is terrible",
    "Great film, highly recommended",
    "Waste of time, don't watch",
    "Amazing performance by the actors"
]

# Create embeddings
sequences, tokenizer = create_word_embeddings(sample_texts, max_words=100, max_len=10)

print("Tokenized Sequences:")
for i, seq in enumerate(sequences):
    print(f"Text {i+1}: {seq}")

print(f"\nVocabulary size: {len(tokenizer.word_index)}")
print(f"Word index: {tokenizer.word_index}")
```

## Language Models

### Basic Language Model

```python
def create_language_model(vocab_size, embedding_dim=128, lstm_units=256):
    """
    Create a basic language model
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension
        lstm_units: Number of LSTM units
    
    Returns:
        Language model
    """
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(lstm_units),
        layers.Dropout(0.2),
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_sequences_for_language_model(texts, seq_length=10):
    """
    Prepare sequences for language model training
    
    Args:
        texts: List of text strings
        seq_length: Length of input sequences
    
    Returns:
        Input sequences and target words
    """
    # Tokenize all texts
    all_words = []
    for text in texts:
        words = word_tokenize(text.lower())
        all_words.extend(words)
    
    # Create word to index mapping
    unique_words = list(set(all_words))
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    index_to_word = {i: word for word, i in word_to_index.items()}
    
    # Create sequences
    input_sequences = []
    target_words = []
    
    for i in range(len(all_words) - seq_length):
        input_seq = all_words[i:i + seq_length]
        target_word = all_words[i + seq_length]
        
        input_sequences.append([word_to_index[word] for word in input_seq])
        target_words.append(word_to_index[target_word])
    
    return np.array(input_sequences), np.array(target_words), word_to_index, index_to_word

# Example usage
sample_texts = [
    "Natural language processing is amazing",
    "Machine learning helps computers understand text",
    "Deep learning models can generate text",
    "Text analysis is important for NLP"
]

# Prepare sequences
input_sequences, target_words, word_to_index, index_to_word = prepare_sequences_for_language_model(sample_texts, seq_length=3)

print(f"Input sequences shape: {input_sequences.shape}")
print(f"Target words shape: {target_words.shape}")
print(f"Vocabulary size: {len(word_to_index)}")
```

## Text Classification

### Traditional Text Classification

```python
def create_text_classifier(texts, labels, method='tfidf'):
    """
    Create text classifier using traditional methods
    
    Args:
        texts: List of text strings
        labels: List of labels
        method: 'tfidf' or 'bow'
    
    Returns:
        Trained classifier and vectorizer
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Create vectorizer
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    else:
        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    # Transform texts
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)
    
    # Train classifier
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train_vectors, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_vectors)
    accuracy = classifier.score(X_test_vectors, y_test)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return classifier, vectorizer, accuracy

# Example usage
sample_texts = [
    "I love this product, it's amazing!",
    "This is terrible, worst purchase ever",
    "Great quality, highly recommended",
    "Don't buy this, waste of money",
    "Excellent service and fast delivery",
    "Very disappointed with the quality",
    "Best product I've ever used",
    "Poor customer service experience"
]

sample_labels = ['positive', 'negative', 'positive', 'negative', 
                'positive', 'negative', 'positive', 'negative']

classifier, vectorizer, accuracy = create_text_classifier(sample_texts, sample_labels, method='tfidf')
```

### Deep Learning Text Classification

```python
def create_deep_text_classifier(texts, labels, max_words=1000, max_len=100):
    """
    Create deep learning text classifier
    
    Args:
        texts: List of text strings
        labels: List of labels
        max_words: Maximum number of words
        max_len: Maximum sequence length
    
    Returns:
        Trained model and tokenizer
    """
    # Prepare data
    sequences, tokenizer = create_word_embeddings(texts, max_words, max_len)
    
    # Convert labels to numerical
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    numerical_labels = label_encoder.fit_transform(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, numerical_labels, test_size=0.2, random_state=42
    )
    
    # Create model
    vocab_size = len(tokenizer.word_index) + 1
    num_classes = len(label_encoder.classes_)
    
    model = keras.Sequential([
        layers.Embedding(vocab_size, 128, input_length=max_len),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return model, tokenizer, label_encoder, history

# Example usage
model, tokenizer, label_encoder, history = create_deep_text_classifier(
    sample_texts, sample_labels
)
```

## Sentiment Analysis

### Basic Sentiment Analysis

```python
def perform_sentiment_analysis(texts):
    """
    Perform sentiment analysis using transformers
    
    Args:
        texts: List of text strings
    
    Returns:
        Sentiment analysis results
    """
    # Load sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")
    
    # Analyze sentiments
    results = []
    for text in texts:
        result = sentiment_pipeline(text)
        results.append(result[0])
    
    return results

def create_sentiment_classifier(texts, sentiments):
    """
    Create custom sentiment classifier
    
    Args:
        texts: List of text strings
        sentiments: List of sentiment labels
    
    Returns:
        Trained sentiment classifier
    """
    # Preprocess texts
    processed_texts = [comprehensive_text_preprocessing(text) for text in texts]
    
    # Create TF-IDF features
    tfidf_matrix, vocabulary = create_tfidf_representation(processed_texts)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, sentiments, test_size=0.2, random_state=42
    )
    
    # Train classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    
    # Evaluate
    accuracy = classifier.score(X_test, y_test)
    y_pred = classifier.predict(X_test)
    
    print(f"Sentiment Analysis Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return classifier, tfidf_matrix

# Example usage
sentiment_texts = [
    "I absolutely love this product!",
    "This is the worst thing I've ever bought",
    "It's okay, nothing special",
    "Amazing quality and great value",
    "Terrible customer service"
]

sentiment_labels = ['positive', 'negative', 'neutral', 'positive', 'negative']

# Using transformers
transformer_results = perform_sentiment_analysis(sentiment_texts)
print("Transformer Results:")
for text, result in zip(sentiment_texts, transformer_results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}, Confidence: {result['score']:.4f}")
    print()

# Using custom classifier
classifier, tfidf_matrix = create_sentiment_classifier(sentiment_texts, sentiment_labels)
```

## Named Entity Recognition

### Basic NER

```python
def perform_ner(texts):
    """
    Perform Named Entity Recognition
    
    Args:
        texts: List of text strings
    
    Returns:
        NER results
    """
    # Load NER pipeline
    ner_pipeline = pipeline("ner")
    
    # Perform NER
    results = []
    for text in texts:
        result = ner_pipeline(text)
        results.append(result)
    
    return results

def extract_entities_from_text(text):
    """
    Extract named entities using NLTK
    
    Args:
        text: Input text string
    
    Returns:
        Dictionary of entities by type
    """
    # Tokenize and tag
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    
    # Extract named entities
    named_entities = ne_chunk(pos_tags)
    
    # Organize entities by type
    entities = {}
    for chunk in named_entities:
        if hasattr(chunk, 'label'):
            entity_type = chunk.label()
            entity_text = ' '.join([word for word, tag in chunk.leaves()])
            
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(entity_text)
    
    return entities

# Example usage
ner_texts = [
    "Apple Inc. is headquartered in Cupertino, California.",
    "John Smith works at Google in New York.",
    "The Eiffel Tower is located in Paris, France."
]

# Using transformers
transformer_ner = perform_ner(ner_texts)
print("Transformer NER Results:")
for text, entities in zip(ner_texts, transformer_ner):
    print(f"Text: {text}")
    for entity in entities:
        print(f"  {entity['entity']}: {entity['word']} ({entity['score']:.4f})")
    print()

# Using NLTK
for text in ner_texts:
    entities = extract_entities_from_text(text)
    print(f"Text: {text}")
    for entity_type, entity_list in entities.items():
        print(f"  {entity_type}: {entity_list}")
    print()
```

## Text Generation

### Basic Text Generation

```python
def create_text_generator(vocab_size, embedding_dim=128, lstm_units=256):
    """
    Create text generation model
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension
        lstm_units: Number of LSTM units
    
    Returns:
        Text generation model
    """
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(lstm_units),
        layers.Dropout(0.2),
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_text(model, tokenizer, seed_text, max_length=50, temperature=1.0):
    """
    Generate text using trained model
    
    Args:
        model: Trained text generation model
        tokenizer: Fitted tokenizer
        seed_text: Starting text
        max_length: Maximum length of generated text
        temperature: Sampling temperature
    
    Returns:
        Generated text
    """
    # Tokenize seed text
    seed_tokens = tokenizer.texts_to_sequences([seed_text])[0]
    
    generated_text = seed_text
    
    for _ in range(max_length):
        # Prepare input
        input_seq = np.array([seed_tokens])
        
        # Predict next word
        predictions = model.predict(input_seq, verbose=0)
        
        # Apply temperature
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        preds = exp_preds / np.sum(exp_preds)
        
        # Sample next word
        predicted_id = np.random.choice(len(preds[0]), p=preds[0])
        
        # Add to generated text
        for word, index in tokenizer.word_index.items():
            if index == predicted_id:
                generated_text += " " + word
                break
        
        # Update seed tokens
        seed_tokens.append(predicted_id)
        seed_tokens = seed_tokens[1:]
    
    return generated_text

# Example usage
generation_texts = [
    "Natural language processing is a field of artificial intelligence",
    "Machine learning algorithms can process and understand text",
    "Deep learning models have revolutionized the way we handle language"
]

# Prepare data for text generation
sequences, tokenizer = create_word_embeddings(generation_texts, max_words=100, max_len=10)

# Create and train model
vocab_size = len(tokenizer.word_index) + 1
generator_model = create_text_generator(vocab_size)

# Note: In practice, you would train this model with proper sequences
# For demonstration, we'll just show the model structure
print("Text Generation Model:")
generator_model.summary()
```

## Best Practices

### Model Evaluation

```python
def evaluate_nlp_model(model, X_test, y_test, model_type='traditional'):
    """
    Evaluate NLP model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_type: Type of model
    
    Returns:
        Evaluation results
    """
    if model_type == 'traditional':
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
    else:
        y_pred = np.argmax(model.predict(X_test), axis=1)
        accuracy = np.mean(y_pred == y_test)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'confusion_matrix': cm,
        'classification_report': report
    }

def visualize_text_analysis(texts, labels):
    """
    Visualize text analysis results
    
    Args:
        texts: List of text strings
        labels: List of labels
    """
    # Text length analysis
    text_lengths = [len(text.split()) for text in texts]
    
    plt.figure(figsize=(15, 5))
    
    # Text length distribution
    plt.subplot(1, 3, 1)
    plt.hist(text_lengths, bins=20, alpha=0.7)
    plt.title('Text Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    
    # Label distribution
    plt.subplot(1, 3, 2)
    label_counts = pd.Series(labels).value_counts()
    plt.bar(label_counts.index, label_counts.values)
    plt.title('Label Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    
    # Average text length by label
    plt.subplot(1, 3, 3)
    df = pd.DataFrame({'text': texts, 'label': labels, 'length': text_lengths})
    avg_lengths = df.groupby('label')['length'].mean()
    plt.bar(avg_lengths.index, avg_lengths.values)
    plt.title('Average Text Length by Label')
    plt.xlabel('Labels')
    plt.ylabel('Average Length')
    
    plt.tight_layout()
    plt.show()

# Example usage
evaluation_results = evaluate_nlp_model(classifier, X_test_vectors, y_test, 'traditional')
visualize_text_analysis(sample_texts, sample_labels)
```

### Performance Optimization

```python
def optimize_nlp_pipeline():
    """
    Optimize NLP pipeline for performance
    
    Returns:
        Optimized pipeline components
    """
    # Use efficient tokenization
    def efficient_tokenize(texts):
        return [text.lower().split() for text in texts]
    
    # Use efficient vectorization
    def efficient_vectorize(texts, max_features=1000):
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 1),  # Use only unigrams for speed
            max_df=0.95,  # Remove very common words
            min_df=2      # Remove very rare words
        )
        return vectorizer.fit_transform(texts)
    
    # Use efficient preprocessing
    def efficient_preprocess(text):
        # Simple and fast preprocessing
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join(text.split())
    
    return {
        'efficient_tokenize': efficient_tokenize,
        'efficient_vectorize': efficient_vectorize,
        'efficient_preprocess': efficient_preprocess
    }

def create_efficient_nlp_model():
    """
    Create efficient NLP model
    
    Returns:
        Efficient model
    """
    # Use lightweight model
    model = keras.Sequential([
        layers.Embedding(1000, 64, input_length=100),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Example usage
optimized_pipeline = optimize_nlp_pipeline()
efficient_model = create_efficient_nlp_model()

print("Optimized NLP Pipeline Created")
print("Efficient Model Summary:")
efficient_model.summary()
```

## Exercises

1. **Text Classification**: Build a classifier to categorize news articles into different topics.
2. **Sentiment Analysis**: Create a sentiment analysis model for product reviews.
3. **Named Entity Recognition**: Extract named entities from a corpus of text.
4. **Text Generation**: Train a language model to generate creative text.
5. **Text Summarization**: Implement extractive text summarization.

## Next Steps

After mastering NLP, explore:
- [Computer Vision](computer_vision_guide.md)
- [Reinforcement Learning](reinforcement_learning_guide.md)
- [Advanced Deep Learning Techniques](../PyTorch/advanced_pytorch_techniques_guide.md)
- [Model Deployment](model_deployment_guide.md) 