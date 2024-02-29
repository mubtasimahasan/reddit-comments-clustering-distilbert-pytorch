from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from data.data_processing import get_data, preprocess_text, tokenize_data, create_dataset
from models.model_training import train_model, test_model, predict_from_test_set, compute_metrics
from utils.utility import set_seed, perform_cross_validation, merge_datasets
from utils.config import CFG

def main():
    # Set seed
    set_seed(CFG.seed)
    # Get data
    (train_texts, valid_texts, test_texts, train_labels, valid_labels, test_labels), target_names = get_data()
    # Preprocess text
    train_texts = [preprocess_text(text) for text in train_texts]
    valid_texts = [preprocess_text(text) for text in valid_texts]
    test_texts = [preprocess_text(text) for text in test_texts]
    # Tokenize data
    tokenizer = DistilBertTokenizerFast.from_pretrained(CFG.model_name)
    train_encodings = tokenize_data(tokenizer, train_texts)
    valid_encodings = tokenize_data(tokenizer, valid_texts)
    test_encodings = tokenize_data(tokenizer, test_texts)
    # Create datasets
    train_dataset = create_dataset(train_encodings, train_labels)
    valid_dataset = create_dataset(valid_encodings, valid_labels)
    test_dataset = create_dataset(test_encodings, test_labels)
    # Train model
    model = DistilBertForSequenceClassification.from_pretrained(CFG.model_name, num_labels=CFG.num_labels).to(CFG.device)    
    trainer = train_model(model, train_dataset, valid_dataset)
    # Test model
    test_model(trainer, test_dataset)
    # Predict from random test data
    predict_from_test_set(trainer, test_dataset, tokenizer, num_samples=3)
    # Perform 5-fold cross-validation
    merged_dataset = merge_datasets(train_dataset, valid_dataset)
    perform_cross_validation(model, merged_dataset, n_splits=5)

if __name__ == "__main__":
    main()