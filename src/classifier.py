from typing import List

import torch
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from typing import List
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change the signature
    of these methods
     """



    ############################################# complete the classifier class below
    
    def __init__(self):
        """
        This should create and initilize the model. Does not take any arguments.
        
        """
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        self.label_encoder = LabelEncoder()
        self.aspect_encoder = OneHotEncoder()
    
    
    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        train_df = pd.read_csv(train_filename, sep='\t', header=None)
        dev_df = pd.read_csv(dev_filename, sep='\t', header=None)

        train_labels = self.label_encoder.fit_transform(train_df[0])
        dev_labels = self.label_encoder.transform(dev_df[0])

        train_aspect = self.aspect_encoder.fit_transform(train_df[1].values.reshape(-1, 1)).toarray()
        dev_aspect = self.aspect_encoder.transform(dev_df[1].values.reshape(-1, 1)).toarray()

        train_encodings = self.tokenizer(train_df[4].tolist(), truncation=True, padding=True)
        dev_encodings = self.tokenizer(dev_df[4].tolist(), truncation=True, padding=True)

        train_context_encodings = self.tokenizer(train_df[3].tolist(), truncation=True, padding=True)
        dev_context_encodings = self.tokenizer(dev_df[3].tolist(), truncation=True, padding=True)

        train_dataset = ClassifierDataset(train_encodings, train_labels, train_aspect, train_context_encodings)
        dev_dataset = ClassifierDataset(dev_encodings, dev_labels, dev_aspect, dev_context_encodings)

        self.model.to(device)
        self.model.train()

        optim = torch.optim.Adam(self.model.parameters(), lr=5e-5)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        for epoch in range(3):
            for batch in train_loader:
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optim.step()


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        self.model.eval()

        test_df = pd.read_csv(data_filename, sep='\t', header=None)
        test_encodings = self.tokenizer(test_df[4].tolist(), truncation=True, padding=True)
        test_dataset = ClassifierDataset(test_encodings)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        all_preds = []

        for batch in test_loader:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())

        return self.label_encoder.inverse_transform(all_preds)
    
class ClassifierDataset(Dataset):
    def __init__(self, encodings, labels=None, aspects=None, context_encodings=None):
        self.encodings = encodings
        self.labels = labels
        self.aspects = aspects
        self.context_encodings = context_encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        if self.aspects:
            item['aspects'] = torch.tensor(self.aspects[idx])
        if self.context_encodings:
            item['context'] = torch.tensor(self.context_encodings[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)





