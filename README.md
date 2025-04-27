# Disaster Tweet Classification with Google Gemini API

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Jupyter%20Notebook-lightgrey)
![Status](https://img.shields.io/badge/status-WIP-yellow)

This project explores text classification using traditional machine learning models and large language models using the Google Gemini API. It aims to classify tweets into categories such as natural disasters (e.g., earthquake, flood, hurricane) or non-disaster-related content.

## Project Overview (Traditional ML)

1. **Data**: The project uses the "Disaster Tweet Corpus 2020," a dataset of human-labeled tweets covering various disaster types, with an equal split of disaster-related and non-disaster-related tweets.
2. **Preprocessing**: Tweets are extensively cleaned and normalized, including removing URLs, special characters, and stopwords, followed by tokenization and lemmatization.
3. **Baseline Evaluation**: A Naive Bayes classifier with a Bag-of-Words (BoW) approach is implemented as the baseline, achieving a macro-averaged F1-score of 0.90.
4. **Improved Models**:
   - Naive Bayes with TF-IDF: Incorporates TF-IDF vectorization but slightly underperforms compared to the baseline.
   - Support Vector Classifier (SVC): Achieves significant improvement using TF-IDF preprocessing.
   - Hyperparameter-Tuned SVC: Further optimization using grid search yields the best performance with a macro-averaged F1-score of 0.98.

## Key Features

- **Baseline and Advanced Models**: Establishes a baseline with Naive Bayes and improves performance with SVC and hyperparameter tuning.

- **Preprocessing**: Demonstrates the impact of preprocessing techniques like TF-IDF and lemmatization on model performance.

- **Evaluation**: Uses confusion matrices and classification reports to assess accuracy, precision, recall, and F1-scores.

## Project Overview (LLM)

1. **Data**: The project uses the "Disaster Tweet Corpus 2020," a dataset of human-labeled tweets covering various disaster types.
2. **Preprocessing**: Tweets are cleaned and normalized for input into the model.
3. **Baseline Evaluation**: The Gemini API's pre-trained model is evaluated using zero-shot prompting and refined system instructions.
4. **Fine-Tuning**: A custom model is tuned using parameter-efficient fine-tuning (PEFT) to improve classification accuracy and reduce token usage.
5. **Evaluation**: The tuned model is tested on a subset of the dataset, comparing accuracy and token efficiency with the baseline.

## Key Features

- **Fine-Tuning**: Demonstrates how to fine-tune a Gemini model for text classification.
- **Prompt Engineering**: Explores techniques to improve model responses using system instructions.
- **Efficiency**: Highlights token savings and cost-effectiveness of tuned models.

## How to Run (LLM classifier notebook)

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/google_gen_ai.git
   cd google_gen_ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment:
   - Create a `.env` file in the project root.
   - Add your Google Gemini API key:
     ```
     GEMINI_API_KEY=your-api-key-here
     ```

4. Run the notebook `llm_classifier.ipynb` to preprocess data, fine-tune the model, and evaluate results.

## Results

- **Baseline Accuracy**: Evaluated using zero-shot prompting.
- **Tuned Model Accuracy**: Improved classification accuracy with reduced token usage.

## Future Work

- Experiment with additional hyperparameter tuning.
- Explore alternative preprocessing techniques.
- Evaluate the model on larger datasets.

## References

- [Google Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Disaster Tweet Corpus 2020](https://doi.org/10.5281/zenodo.3713920)

## Disclaimer

This project uses the Google Gemini API, which is subject to Google's [Terms of Service](https://ai.google.dev/terms). Use of the API and associated services is governed by those terms, and not by this repository's license.
