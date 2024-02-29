import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments

from utils.config import CFG

def train_model(model, train_dataset, valid_dataset):
    # Create Trainer
    training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=CFG.n_epochs,
    per_device_train_batch_size=CFG.batch_size,
    per_device_eval_batch_size=CFG.batch_size,
    warmup_steps=CFG.warmup_steps,
    weight_decay=CFG.weight_decay,
    logging_steps=CFG.logging_steps,
    save_steps=CFG.save_steps,
    evaluation_strategy=CFG.evaluation_strategy,
    load_best_model_at_end=CFG.load_best_model_at_end,
    report_to= CFG.report_to,
    max_steps=CFG.max_steps
    )
    
    trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    )
    
    # Train model
    trainer.train()
    #trainer.save_model()
    
    eval_result = trainer.evaluate()
    print("Evaluation result:", eval_result)
    plot_loss_curves(trainer)
        
    return trainer    

def test_model(trainer, test_dataset):
    eval_result = trainer.evaluate(test_dataset)
    print("Test set evaluation result:", eval_result)
    plot_confusion_matrix(trainer, test_dataset)

def predict_from_test_set(trainer, test_dataset, tokenizer, num_samples):
    # Sample some sentences from the test set
    sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    sampled_data = [test_dataset[i] for i in sample_indices]

    # Iterate over the sampled data and make predictions
    for data in sampled_data:
        inputs = {k: v.clone().detach().unsqueeze(0).to(CFG.device) for k, v in data.items()}
        outputs = trainer.model(**inputs)  # Use trainer.model instead of model
        probs = outputs.logits.softmax(1)
        predicted_class = probs.argmax().item()
        
        # Display the sample text, ground truth, and prediction
        print("Text:", tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        print("Ground Truth:", data["labels"].item())
        print("Predicted Class:", predicted_class)
        print("=" * 50)

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(pred.label_ids, preds)
    f1 = f1_score(pred.label_ids, preds, average='macro')
    return {'accuracy': acc, 'f1_macro': f1}

def plot_loss_curves(trainer):
    train_loss = [item['loss'] for item in trainer.state.log_history if 'loss' in item]
    val_loss = [item['eval_loss'] for item in trainer.state.log_history if 'eval_loss' in item]
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()  
    plt.close()  

def plot_confusion_matrix(trainer, test_dataset):
    preds = trainer.predict(test_dataset)
    pred_labels = np.argmax(preds.predictions, axis=1)
    true_labels = test_dataset.labels  
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(CFG.num_labels))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix (Test Dataset)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    plt.close()  
