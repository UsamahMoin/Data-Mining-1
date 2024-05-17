
# Data Mining Project 1

This project consists of two main components: a Jupyter Notebook and a Python script.

## Files

1. `P1 Intro.ipynb`
2. `UsamahMoinMohammed.py`

### 1. P1 Intro.ipynb

This Jupyter Notebook is an introduction to a project related to text processing and analysis. The notebook includes the following sections:

- **Text Preprocessing**: Tokenization, stop-word removal, and stemming.
- **TF-IDF Calculation**: Calculation of Term Frequency-Inverse Document Frequency for the given documents.
- **Query Processing**: Handling user queries and calculating relevance scores using cosine similarity.
- **Results Display**: Output of the TF-IDF values for specific terms and query results.

#### Usage

To run the notebook:

1. Open the notebook in JupyterLab or Jupyter Notebook.
2. Execute the cells sequentially to preprocess the text, compute TF-IDF values, and handle queries.

### 2. UsamahMoinMohammed.py

This Python script performs text analysis on a collection of documents. It includes the following functionalities:

- **Text Tokenization**: Breaking down text into tokens.
- **Stop-word Removal**: Removing common words that do not contribute to the analysis.
- **Stemming**: Reducing words to their root forms.
- **TF-IDF Computation**: Calculating TF-IDF values for the tokens in the documents.
- **Query Handling**: Processing user queries and computing relevance scores.

#### Key Functions

- `tokenize(text)`: Tokenizes the input text.
- `remove_stop_words(tokens)`: Removes stop-words from the tokenized text.
- `stem(tokens)`: Applies stemming to the tokens.
- `compute_tf(document)`: Computes term frequency for the tokens in a document.
- `compute_idf(documents)`: Computes inverse document frequency across a collection of documents.
- `compute_weights(documents)`: Computes TF-IDF weights for the tokens in the documents.
- `getweight(doc, term)`: Retrieves the weight of a term in a specific document.
- `query(q)`: Processes a query and returns the most relevant document along with its score.

#### Usage

To run the script:

1. Ensure that Python 3.x and the required libraries (such as `nltk` and `math`) are installed.
2. Place the script in the same directory as your text documents.
3. Execute the script in a terminal or command prompt:
   ```sh
   python UsamahMoinMohammed.py
   ```

## Requirements

- Python 3.x
- Jupyter Notebook (for `.ipynb` file)
- NLTK library

## Installation

Install the necessary libraries using pip:

```sh
pip install nltk jupyter
```

## Acknowledgments

- The text processing techniques used in this project are inspired by common practices in natural language processing and information retrieval.
- Thanks to the NLTK library for providing tools for text processing.
