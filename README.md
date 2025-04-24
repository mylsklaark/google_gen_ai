# Disaster Tweet Classification with Google Gemini API

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Jupyter%20Notebook-lightgrey)
![Status](https://img.shields.io/badge/status-WIP-yellow)

This project explores text classification using the Google Gemini API. It fine-tunes a Gemini model to classify tweets into categories such as natural disasters (e.g., earthquake, flood, hurricane) or non-disaster-related content.

## Project Overview

1. **Data**: The project uses the "Disaster Tweet Corpus 2020," a dataset of human-labeled tweets covering various disaster types.
2. **Preprocessing**: Tweets are cleaned and normalized for input into the model.
3. **Baseline Evaluation**: The Gemini API's pre-trained model is evaluated using zero-shot prompting and refined system instructions.
4. **Fine-Tuning**: A custom model is tuned using parameter-efficient fine-tuning (PEFT) to improve classification accuracy and reduce token usage.
5. **Evaluation**: The tuned model is tested on a subset of the dataset, comparing accuracy and token efficiency with the baseline.

## Key Features

- **Fine-Tuning**: Demonstrates how to fine-tune a Gemini model for text classification.
- **Prompt Engineering**: Explores techniques to improve model responses using system instructions.
- **Efficiency**: Highlights token savings and cost-effectiveness of tuned models.

## How to Run

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
