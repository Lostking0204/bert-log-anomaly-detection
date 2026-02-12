import torch
import torch.nn as nn
from transformers import BertModel
from torch.optim import AdamW

class BERTLogClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        super(BERTLogClassifier, self).__init__()
        # 1. Pre-trained BERT Encoder (12 layers, 110M parameters) [cite: 407, 509]
        self.bert = BertModel.from_pretrained(model_name)
        
        # 2. Fully Connected Layer (Classifier Head) [cite: 460, 484]
        # Maps the 768-dimensional [CLS] vector to 2 classes (Normal/Anomalous)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        # Obtain semantic representation from BERT [cite: 438, 477]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # BERT-Log uses the [CLS] token vector at index 0 [cite: 438]
        cls_output = outputs.last_hidden_state[:, 0, :] 
        
        # Pass through linear classifier for anomaly prediction [cite: 481, 482]
        logits = self.classifier(cls_output)
        return logits

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTLogClassifier().to(device)
    
    # Optimizer and Learning Rate from BERT-Log research 
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- Model Initialized ---")
    print(f"Device: {device}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # Testing the forward pass with a dummy batch
    print("Testing forward pass...")
    test_ids = torch.randint(0, 30522, (2, 20)).to(device)
    test_mask = torch.ones((2, 20)).to(device)
    output = model(test_ids, test_mask)
    print(f"Output Logits Shape: {output.shape} (Batch size, Num labels)")
