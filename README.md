# Vector Space Models and Word Embeddings

## Overview
This project explores different ways of representing text as vectors, including sparse word vectors that model a word’s context and off-the-shelf **Word2Vec embeddings**. We also experiment with **Skip-gram** training, explore **word analogies**, and evaluate embeddings using the **TOEFL synonym test**.

The primary tasks are constructing word vectors using a context-window-based approach, performing manual steps of Word2Vec training with negative sampling, and evaluating embeddings for their ability to learn semantic relationships between words.

## Project Structure
- **Data**: 
  - A subset of the Brown Corpus (`brown100k.txt`) is used to train sparse word vectors.
  - Pre-trained Word2Vec vectors are used for evaluation.
  - Additional evaluation datasets include the **analogies.txt** and **TOEFL dataset (toefl.txt)**.
  
- **Main Concepts**:
  - Vector representations of words (sparse vectors and pre-trained Word2Vec)
  - Skip-Gram training with negative sampling
  - Analogy testing and TOEFL synonym evaluation

## Tasks

### Part 1: Train Sparse Word Vectors
- **Objective**: Construct term-term matrices based on a context-window approach with a window size of `k=2` using the Brown Corpus subset.
- **Procedure**:
  1. Load the Brown corpus (`brown100k.txt`), convert words to lowercase, and create a vocabulary of the top 1000 most frequent words.
  2. Construct a term-term matrix where each row represents a word, and columns represent the context words within a window of `k=2`.
 
### Part 2: Skip-Grams and Negative Sampling
- **Objective**: Perform a manual step of training a Word2Vec model with skip-gram and negative sampling using a toy example sentence: `"Cat litter smells bad"`.
- **Steps**:
  1. Initialize word embeddings for the words "cat", "litter", "remote", and "oatmeal" in a 2-dimensional space.
  2. Calculate the initial loss function and update the vectors using stochastic gradient descent (SGD).
  3. Recalculate the loss function and plot the updated vectors.
  
### Part 3: Evaluating Word2Vec

#### Part 3a: Word Analogies
- **Objective**: Test the performance of Word2Vec embeddings on word analogies using the analogy dataset (`analogies.txt`).
- **Procedure**:
  - Implement analogy solving by calculating vector differences, e.g., `vector_embedding(Greece) - vector_embedding(Athens) + vector_embedding(Baghdad)`.
  - Report accuracy based on different types of analogies (e.g., capitals of countries, currency, city-in-state).

#### Part 3b: TOEFL Synonym Test
- **Objective**: Evaluate word embeddings using the TOEFL synonym dataset (`toefl.txt`), a multiple-choice test designed to assess knowledge of synonyms.
- **Procedure**:
  - For each synonym question, calculate the cosine similarity between the target word and the choices.
  - Output the answer with the highest similarity and report the model’s overall accuracy.
  
## Requirements
- Python 3.x
- Libraries:
  - `scipy`
  - `sklearn`
  - `gensim`
  - `matplotlib`
  
## How to Run
1. Install the required packages
2. Load the datasets and preprocess the Brown Corpus.
3. Train the sparse word vectors and perform skip-gram updates.
4. Evaluate the word embeddings on the analogy dataset and TOEFL synonym test.

## Results
- **Sparse Vector Training**: 
  - Term-term matrix constructed using the Brown corpus.
  - Nearest words to a chosen target word using cosine similarity.
- **Skip-Gram Updates**: 
  - Initial and updated word vectors plotted with loss function updates.
  - Changes in model predictions after one step of SGD.
- **Word2Vec Evaluation**:
  - Accuracy on word analogies across different categories.
  - Performance on the TOEFL synonym test, including accuracy and error analysis.

## References
1. Python File Open. (2019). W3schools.com. https://www.w3schools.com/python/python_file_open.asp

2. How to count the occurrences of a list item? (n.d.). Www.tutorialsteacher.com. Retrieved from https://www.tutorialsteacher.com/articles/how-to-count-occurences-of-list-items-in-python

3. Gern Blanston. (2009, March 5). How do I sort a dictionary by value? Stack Overflow. https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value

4. Python | Get dictionary keys as a list. (2019, January 28). GeeksforGeeks. https://www.geeksforgeeks.org/python-get-dictionary-keys-as-a-list/

5. Efficiently count zero elements in numpy array? (n.d.). Stack Overflow. Retrieved from https://stackoverflow.com/questions/42916330/efficiently-count-zero-elements-in-numpy-array

6. sklearn.metrics.pairwise.cosine_similarity. (n.d.). Scikit-Learn. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html

7. How do I get indices of N maximum values in a NumPy array? (n.d.). Stack Overflow. https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array

8. numpy.fill_diagonal — NumPy v1.26 Manual. (n.d.). Numpy.org. Retrieved from https://numpy.org/doc/stable/reference/generated/numpy.fill_diagonal.html

9. NLTK :: Sample usage for gensim. (n.d.). Www.nltk.org. https://www.nltk.org/howto/gensim.html

10. How to get the dimensions of a word2vec vector? (n.d.). Stack Overflow. Retrieved from https://stackoverflow.com/questions/71792841/how-to-get-the-dimensions-of-a-word2vec-vector

11. how to split a text file into multiple list based on whitespacing in python? (n.d.). Stack Overflow. Retrieved from https://stackoverflow.com/questions/28018285/how-to-split-a-text-file-into-multiple-list-based-on-whitespacing-in-python

12. Python Multiprocessing Example | DigitalOcean. (n.d.). Www.digitalocean.com. https://www.digitalocean.com/community/tutorials/python-multiprocessing-example

13. How to check if a key exists in a word2vec trained model or not. (n.d.). Stack Overflow. Retrieved from https://stackoverflow.com/questions/30301922/how-to-check-if-a-key-exists-in-a-word2vec-trained-model-or-not


14. python - Scatter plot with different text at each data point. (n.d.). Stack Overflow. https://stackoverflow.com/questions/14432557/scatter-plot-with-different-text-at-each-data-point

15. How to calculate a logistic sigmoid function in Python? (n.d.). Stack Overflow. Retrieved from https://stackoverflow.com/questions/3985619/how-to-calculate-a-logistic-sigmoid-function-in-python

