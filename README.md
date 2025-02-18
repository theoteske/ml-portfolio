# Machine Learning Portfolio

This repository showcases a collection of machine learning projects, focusing on deep learning, natural language processing, and neural network implementations. Each project demonstrates different aspects of modern machine learning techniques and architectures.

## Projects

### Sentiment Analysis with Multiple Approaches
Implementation of three different approaches to sentiment analysis on the IMDB dataset:
- TF-IDF + Logistic Regression pipeline achieving 89% accuracy
- Custom RNN with LSTM layers and embeddings built in PyTorch
- Fine-tuned DistilBERT model reaching 92% accuracy on the test set

### Language Modeling with RNN
Character-level language model trained on "The Mysterious Island" by Jules Verne:
- Implementation of LSTM-based RNN in PyTorch
- Autoregressive text generation capturing the style of the source text
- Temperature-controlled sampling for output diversity

### Neural Networks with Noise Injection
Investigation of noise injection in neural networks for XOR classification:
- Custom implementation of noise injection layer
- Demonstration of relationship between noise and Tikhonov regularization
- Achieved 98% accuracy on test set
- Based on research by [Bishop (1995)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-tikhonov-nc-95.pdf)

### Convolutional Neural Networks
Two CNN implementations focusing on different computer vision tasks:
- MNIST digit classification achieving 99% accuracy
- CelebA smile detection with data augmentation
- Implementation of various image transformation techniques

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-portfolio.git

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

Each project is contained in its own Jupyter notebook with detailed explanations and visualizations. HTML versions of the notebooks are also provided for easy viewing.
Navigate to the notebooks directory and open any .ipynb file in Jupyter Notebook or view the corresponding .html file directly in your browser.

## Technologies

- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- NLTK
- TorchText
- NumPy
- Pandas
- Matplotlib