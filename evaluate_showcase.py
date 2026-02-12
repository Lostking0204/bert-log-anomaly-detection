import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
# Direct imports since files are in the current root
from log_classifier import BERTLogClassifier
from data_loader import prepare_batch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Initialize and load the model
    model = BERTLogClassifier().to(device)
    model.load_state_dict(torch.load('bert_log_model.pth', map_location=device))
    model.eval()

    print("Loading pre-processed sequences for final showcase evaluation...")
    seq_df = pd.read_pickle('hdfs_sequences.pkl')
    labels_df = pd.read_csv('anomaly_label.csv')
    labels_df.columns = ['BlockID', 'Label']
    labels_df['Label'] = labels_df['Label'].map({'Normal': 0, 'Anomaly': 1})
    
    # Merge and sample for evaluation
    eval_data = pd.merge(seq_df, labels_df, on='BlockID', how='inner')
    eval_sample = eval_data.sample(n=50000, random_state=42)
    
    dataset = prepare_batch(tokenizer, eval_sample['EventID'].tolist(), eval_sample['Label'].tolist())
    loader = DataLoader(dataset, batch_size=64)

    all_preds, all_labels, all_probs = [], [], []
    print(f"Evaluating 50,000 sequences on {device}...")
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate Results
    print(f"\n--- BERT-LOG FINAL SHOWCASE METRICS ---")
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds):.4f}")
    print(f"F1-Score:  {f1_score(all_labels, all_preds):.4f}")

    # Generate Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title('HDFS Anomaly Detection: Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('results/confusion_matrix.png')

    # Generate ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc(fpr, tpr):.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curve.png')
    
    print("\n[✔] Evaluation complete. Proof images saved in 'results/' folder.")

if __name__ == "__main__":
    evaluate()
