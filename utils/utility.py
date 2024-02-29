import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch

from data.data_processing import create_dataset
from models.model_training import train_model
from utils.config import CFG

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def merge_datasets(train_dataset, valid_dataset):
    merged_encodings = {key: np.concatenate([train_dataset.encodings[key], valid_dataset.encodings[key]], axis=0) 
                        for key in train_dataset.encodings.keys()}
    merged_labels = np.concatenate([train_dataset.labels, valid_dataset.labels], axis=0)
    return create_dataset(merged_encodings, merged_labels)

def perform_cross_validation(model, merged_dataset, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG.seed)
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(merged_dataset.encodings['input_ids'], merged_dataset.labels)):
        print("=" * 50)
        print(f"Fold {fold+1}/{n_splits}")
        train_encodings_fold = {key: merged_dataset.encodings[key][train_idx] for key in merged_dataset.encodings.keys()}
        train_labels_fold = merged_dataset.labels[train_idx]
        valid_encodings_fold = {key: merged_dataset.encodings[key][val_idx] for key in merged_dataset.encodings.keys()}
        valid_labels_fold = merged_dataset.labels[val_idx]
        
        train_fold_dataset = create_dataset(train_encodings_fold, train_labels_fold)
        valid_fold_dataset = create_dataset(valid_encodings_fold, valid_labels_fold)
        
        trainer = train_model(model, train_fold_dataset, valid_fold_dataset)
        eval_result = trainer.evaluate(valid_fold_dataset)
        
        fold_accuracy = eval_result['eval_accuracy']
        print(f"Accuracy on validation set for fold {fold+1}: {fold_accuracy}\n")
        fold_accuracies.append(fold_accuracy)
    
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f"Average accuracy over {n_splits} folds: {avg_accuracy}")    

