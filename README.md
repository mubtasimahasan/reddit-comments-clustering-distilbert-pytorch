# Reddit Comments Subreddit Clustering

This project utilizes Natural Language Processing (NLP) techniques to classify topics of Reddit comments. It involves preprocessing the comments, including text cleaning, tokenization, and lemmatization, and then training a transformer-based model for topic classification. The code implements a pipeline that utilizes a DistilBERT-based sequence classification model for training and evaluation.

## Project Structure

- `cli.py`: Main script to execute the project. It orchestrates data processing, model training, evaluation, and cross-validation.
- `configs/config.yaml`: Configuration file containing hyperparameters and settings for the project.
- `data/data_processing.py`: Module for data loading, preprocessing, and dataset creation.
- `data/reddit_data.csv`: Dataset containing Reddit comments and their corresponding topics.
- `models/model_training.py`: Module for model training, testing, and evaluation.
- `utils/config.py`: Configuration class to load settings from the YAML file.
- `utils/utility.py`: Utility functions for setting seed, merging datasets, and performing cross-validation.

## Usage

1. Ensure all required libraries are installed. You can install them using `pip install -r requirements.txt`.
2. Modify the configuration file (`configs/config.yaml`) if necessary to adjust hyperparameters or settings.
3. Run the `cli.py` script to execute the project.

## Notebooks

- [Kaggle_Script_Execution.ipynb](Kaggle_Script_Execution.ipynb): This notebook demonstrates how to clone this project in Kaggle and execute the `cli.py` script. Please note that the script is executed in debug mode, set to true, and runs for 2 epochs.

- [Traditional_Notebook.ipynb](Traditional_Notebook.ipynb): This notebook provides a traditional Jupyter Notebook version of this project, showcasing the entire training and evaluation process.