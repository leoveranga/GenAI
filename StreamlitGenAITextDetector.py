import streamlit as st
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import sys

# Set NLTK data path to include local directory
nltk.data.path.insert(0, '/home/ubuntu/ai_text_detector/nltk_data')

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir='/home/ubuntu/ai_text_detector/nltk_data')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir='/home/ubuntu/ai_text_detector/nltk_data')

# Set page configuration
st.set_page_config(
    page_title="AI Text Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = text.split()  # Simple whitespace tokenization as fallback
    try:
        tokens = word_tokenize(text)
    except:
        pass
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    except:
        # If stopwords fail, just continue with the tokens
        pass
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

# Function to extract features from text
def extract_features(text):
    features = {}
    
    # Word count - simple split as fallback
    words = text.split()
    try:
        words = word_tokenize(text)
    except:
        pass
    
    features['word_count'] = len(words)
    
    # Sentence count - simple heuristic as fallback
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    try:
        sentences = sent_tokenize(text)
    except:
        pass
    
    features['sentence_count'] = len(sentences)
    
    # Average word length
    if len(words) > 0:
        features['avg_word_length'] = sum(len(word) for word in words) / len(words)
    else:
        features['avg_word_length'] = 0
    
    # Average sentence length
    if len(sentences) > 0:
        features['avg_sentence_length'] = len(words) / len(sentences)
    else:
        features['avg_sentence_length'] = 0
    
    # Unique words ratio
    if len(words) > 0:
        features['unique_words_ratio'] = len(set(words)) / len(words)
    else:
        features['unique_words_ratio'] = 0
    
    # Lexical diversity (Type-Token Ratio)
    if len(words) > 0:
        features['lexical_diversity'] = len(set(words)) / len(words)
    else:
        features['lexical_diversity'] = 0
    
    # Sentence complexity (average commas per sentence)
    comma_count = text.count(',')
    if len(sentences) > 0:
        features['sentence_complexity'] = comma_count / len(sentences)
    else:
        features['sentence_complexity'] = 0
    
    return features

# Function to check if text is long enough
def is_text_long_enough(text, min_words=500):
    # Simple split as fallback
    words = text.split()
    try:
        words = word_tokenize(text)
    except:
        pass
    return len(words) >= min_words

# Function to train model (in a real app, this would be pre-trained)
def train_model():
    # This is a simplified example. In a real application, you would:
    # 1. Use a large dataset of human and AI-generated texts
    # 2. Train a more sophisticated model
    # 3. Save the model for future use
    
    # For demonstration purposes, we'll create a simple model with synthetic data
    np.random.seed(42)
    
    # Generate synthetic features for human text
    human_features = np.random.normal(loc=[1000, 50, 5.2, 20, 0.7, 0.7, 0.8], 
                                     scale=[200, 10, 0.3, 3, 0.1, 0.1, 0.2], 
                                     size=(100, 7))
    
    # Generate synthetic features for AI text
    ai_features = np.random.normal(loc=[1000, 40, 4.8, 25, 0.5, 0.5, 0.4], 
                                  scale=[200, 8, 0.2, 4, 0.1, 0.1, 0.1], 
                                  size=(100, 7))
    
    # Combine features and create labels
    X = np.vstack([human_features, ai_features])
    y = np.hstack([np.zeros(100), np.ones(100)])  # 0 for human, 1 for AI
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# Function to predict if text is AI-generated
def predict_ai_text(text, model):
    # Extract features
    features = extract_features(text)
    
    # Convert features to array
    feature_array = np.array([[
        features['word_count'],
        features['sentence_count'],
        features['avg_word_length'],
        features['avg_sentence_length'],
        features['unique_words_ratio'],
        features['lexical_diversity'],
        features['sentence_complexity']
    ]])
    
    # Make prediction
    prediction = model.predict(feature_array)[0]
    probability = model.predict_proba(feature_array)[0][1]  # Probability of being AI-generated
    
    return prediction, probability

# Function to analyze text patterns
def analyze_text_patterns(text):
    analysis = {}
    
    # Word count
    words = text.split()
    try:
        words = word_tokenize(text)
    except:
        pass
    
    analysis['word_count'] = len(words)
    
    # Sentence count
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    try:
        sentences = sent_tokenize(text)
    except:
        pass
    
    analysis['sentence_count'] = len(sentences)
    
    # Average word length
    if len(words) > 0:
        analysis['avg_word_length'] = round(sum(len(word) for word in words) / len(words), 2)
    else:
        analysis['avg_word_length'] = 0
    
    # Average sentence length (in words)
    if len(sentences) > 0:
        analysis['avg_sentence_length'] = round(len(words) / len(sentences), 2)
    else:
        analysis['avg_sentence_length'] = 0
    
    # Unique words ratio
    if len(words) > 0:
        analysis['unique_words_ratio'] = round(len(set(words)) / len(words), 2)
    else:
        analysis['unique_words_ratio'] = 0
    
    # Lexical diversity (Type-Token Ratio)
    if len(words) > 0:
        analysis['lexical_diversity'] = round(len(set(words)) / len(words), 2)
    else:
        analysis['lexical_diversity'] = 0
    
    # Sentence complexity (average commas per sentence)
    comma_count = text.count(',')
    if len(sentences) > 0:
        analysis['sentence_complexity'] = round(comma_count / len(sentences), 2)
    else:
        analysis['sentence_complexity'] = 0
    
    return analysis

# Main app
def main():
    st.title("AI Text Detector")
    st.markdown("### Detect if a text is AI-generated or human-written")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application analyzes text to determine if it was likely "
        "written by a human or generated by an AI system like ChatGPT, "
        "GPT-4, or other large language models."
    )
    st.sidebar.title("Instructions")
    st.sidebar.markdown(
        "1. Enter or paste text in the text area\n"
        "2. The text should be at least 500 words\n"
        "3. Click 'Analyze Text' to get results"
    )
    
    # Text input
    text_input = st.text_area("Enter text (minimum 500 words):", height=300)
    
    # Check if analyze button is clicked
    if st.button("Analyze Text"):
        if not text_input:
            st.error("Please enter some text to analyze.")
        else:
            # Check if text is long enough
            if not is_text_long_enough(text_input):
                st.warning("Text should be at least 500 words for accurate analysis. Please enter a longer text.")
            else:
                with st.spinner("Analyzing text..."):
                    # Preprocess text
                    processed_text = preprocess_text(text_input)
                    
                    # Train model (in a real app, this would be pre-trained)
                    model = train_model()
                    
                    # Make prediction
                    prediction, probability = predict_ai_text(processed_text, model)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Create columns for results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.error("🤖 AI-Generated Text")
                            st.markdown(f"Confidence: **{probability*100:.2f}%**")
                        else:
                            st.success("👤 Human-Written Text")
                            st.markdown(f"Confidence: **{(1-probability)*100:.2f}%**")
                    
                    with col2:
                        # Display probability gauge
                        st.markdown("### AI Probability")
                        st.progress(probability)
                        st.markdown(f"**{probability*100:.2f}%** chance of being AI-generated")
                    
                    # Display text analysis
                    st.subheader("Text Analysis")
                    analysis = analyze_text_patterns(text_input)
                    
                    # Create columns for analysis metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Word Count", analysis['word_count'])
                        st.metric("Average Word Length", analysis['avg_word_length'])
                        st.metric("Unique Words Ratio", analysis['unique_words_ratio'])
                    
                    with col2:
                        st.metric("Sentence Count", analysis['sentence_count'])
                        st.metric("Average Sentence Length", analysis['avg_sentence_length'])
                        st.metric("Lexical Diversity", analysis['lexical_diversity'])
                    
                    with col3:
                        st.metric("Sentence Complexity", analysis['sentence_complexity'])
                    
                    # Explanation
                    st.subheader("How it Works")
                    st.markdown(
                        "This tool analyzes various linguistic features of the text to determine "
                        "if it was likely written by a human or generated by an AI. "
                        "AI-generated text often has different patterns of word usage, "
                        "sentence structure, and complexity compared to human writing."
                    )
                    
                    # Disclaimer
                    st.warning(
                        "**Disclaimer**: This is a simplified demonstration model and may not be 100% accurate. "
                        "The model uses synthetic training data and basic linguistic features. "
                        "A production-ready system would use a larger dataset of real human and AI texts "
                        "and more sophisticated natural language processing techniques."
                    )

if __name__ == "__main__":
    main()
