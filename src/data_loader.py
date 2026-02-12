import torch
from torch.utils.data import Dataset

class LogDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def prepare_batch(tokenizer, sequences, labels):
    """
    Tokenizes sequences and packages them with labels[cite: 381, 395].
    """
    texts = [" ".join(seq) for seq in sequences]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=510)
    return LogDataset(encodings, labels)

if __name__ == "__main__":
    print("Data loader script created. Ready for dataset integration.")
