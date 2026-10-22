# Simple RNN for Text Classification – Sarcasm Detection

## Project Overview
This project uses a **Simple Recurrent Neural Network (RNN)** to classify news headlines as sarcastic or non-sarcastic. The goal is to understand how deep learning and text preprocessing can be used for Natural Language Processing (NLP) classification tasks.

The model learns patterns from headline text and predicts whether a headline contains sarcasm.

## Dataset
- **Dataset:** Sarcasm Headlines Dataset
- **File:** `Sarcasm_Headlines_Dataset.json`
- **Original records:** 26,709
- **Text column:** `headline`
- **Target column:** `is_sarcastic`
- **Target labels:**
  - `0` – Not Sarcastic
  - `1` – Sarcastic

The `article_link` column is excluded because it is not required for the classification task.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Natural Language Processing (NLP)

## Project Workflow

### 1. Data Loading and Inspection
- Load the JSON dataset using Pandas.
- Inspect dataset dimensions, column names, data types, and descriptive statistics.
- Check missing values and target class distribution.

### 2. Data Cleaning
- Select the `headline` and `is_sarcastic` columns.
- Handle missing and empty headline values.
- Standardize headline data types.
- Identify and remove exact duplicate rows.
- Verify duplicate removal after cleaning.

### 3. Exploratory Data Analysis (EDA)
Visualizations include:
- Sarcastic vs. non-sarcastic class distribution.
- Headline character-length distribution.
- Headline length comparison by sarcasm class.
- Word count comparison by class.
- Top frequent words in non-sarcastic headlines.
- Top frequent words in sarcastic headlines.

### 4. Feature Engineering
Two additional features are created:

- **headline_length:** Number of characters in each headline.
- **word_count:** Number of words in each headline.

These features help analyze headline characteristics and compare patterns across classes. They are used for EDA and are not directly passed into the RNN model.

### 5. Outlier Analysis
The Interquartile Range (IQR) method is used to identify potential outliers in headline character length.

Potential outliers are retained because unusually long headlines may still contain meaningful information.

### 6. Text Preprocessing
- Split the data into training and testing sets.
- Use Keras Tokenizer with a vocabulary limit of 5,000 words.
- Use an out-of-vocabulary token for unseen words.
- Convert text into numerical sequences.
- Pad or truncate sequences to a fixed length of 50 tokens.

### 7. Model Architecture
The Simple RNN model consists of the following layers:

1. **Embedding(5000, 32)** – Converts token IDs into 32-dimensional word representations.
2. **SimpleRNN(32)** – Learns sequential patterns in headline text.
3. **Dense(1, activation="sigmoid")** – Produces a probability for binary sarcasm classification.

**Compilation settings:**
- Optimizer: Adam
- Loss function: Binary Crossentropy
- Evaluation metric: Accuracy
- Epochs: 5
- Batch size: 32

### 8. Model Evaluation
The trained model is evaluated on the held-out test dataset.

**Evaluation metric:**
- Test Accuracy

The actual test accuracy is recorded after executing the notebook.

## Key Findings
The dataset analysis shows:

- The original dataset contains 26,709 records.
- After cleaning and exact duplicate removal, 26,602 rows remain.
- Non-sarcastic headlines represent approximately 56.2% of the cleaned dataset.
- Sarcastic headlines represent approximately 43.8%.
- The median headline word count is 10.
- Sarcastic headlines are slightly longer on average in character length, although this alone does not establish predictive importance.
- Some headlines have unusually large character lengths, so potential outliers are retained unless identified as invalid.

## Challenges and Learning Outcomes
- Preparing text data for deep learning models.
- Understanding tokenization and sequence padding.
- Exploring headline characteristics through EDA.
- Applying feature engineering to text datasets.
- Building a recurrent neural network for binary classification.
- Understanding the limitations of Simple RNNs, particularly the vanishing gradient problem.

A plain RNN may struggle to learn long-range dependencies because gradients can become very small during backpropagation through time, making earlier information harder to learn.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   ```

2. Open the notebook in Google Colab or Jupyter Notebook.

3. Upload or configure the dataset path:
   ```python
   /content/Sarcasm_Headlines_Dataset.json
   ```

4. Install dependencies if needed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```

5. Run all notebook cells from top to bottom.

## Conclusion
This project demonstrates an end-to-end NLP classification workflow using a Simple RNN. It combines data cleaning, exploratory analysis, feature engineering, text tokenization, sequence padding, neural network training, and model evaluation to predict sarcasm in news headlines.

## Future Improvements
- Compare Simple RNN performance with LSTM and GRU models.
- Evaluate Precision, Recall, F1-score, and a confusion matrix.
- Experiment with different vocabulary sizes and sequence lengths.
- Apply pretrained word embeddings or transformer-based models.
- Investigate whether engineered text features improve classification performance.

---
**Project:** Sarcasm Headline Classification using Simple RNN  
**Task:** Binary Text Classification  
**Framework:** TensorFlow / Keras
