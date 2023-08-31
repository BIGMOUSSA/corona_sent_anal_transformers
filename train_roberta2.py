from torch.utils.data  import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
#from transformers.optimization import AdamW
from torch.optim import AdamW
import os
#import wandb


config = {
    "model_name" : "roberta-base",
    "max_length" : 80,
    "hidden_state" : 768,
    "csv_fil" : "data/cleaned/nlp_clean.csv",
    "batch_size" : 2,
    "learning_rate" : 2e-5,
    "n_epochs" : 5,
    "n_classes" : 5
}
class MyDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name, max_length):
        self.df = pd.read_csv(csv_file, encoding='ISO-8859-1').iloc[:30]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.df["text"].to_list()
        # Convert numerical labels to one-hot encoded tensors
        num_classes = 5
        #one_hot_labels = torch.zeros(len(self.df), num_classes)
        #one_hot_labels.scatter_(1, torch.tensor(self.df["labels"]).unsqueeze(1), 1)
        #label = one_hot_labels[index]
        label = self.df["labels"].to_list()
        inputs = self.tokenizer(
            text = text,
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_tensors = "pt"
        )


        return {
            "input_ids" : inputs["input_ids"][index],
            "attention_mask" : inputs["attention_mask"][index],
            "labels" : torch.tensor(label[index], dtype=torch.long)
        }

def dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset = dataset,
                      batch_size = batch_size,
                      shuffle = shuffle
                      )


class CustomModel(nn.Module):
    def __init__(self, model_name, n_classes):
        super(CustomModel, self).__init__()
        config = AutoConfig.from_pretrained("roberta-base", num_labels=n_classes) 
        self.pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_name, config = config) #hidden_state 786 Bert_base
        #self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, n_classes)
        #self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, input_ids, attention_mask):
        output = self.pretrained_model(input_ids = input_ids, attention_mask = attention_mask)
        #pooled_output = output.pooler_output
        #output = self.classifier(output.last_hidden_state)
        #output = self.softmax(output)
        #output = self.classifier(pooled_output)
        return output

    



def train_step(model, train_loader, optimizer, loss_fn, device):
    model.train()
    
    total_loss  = 0

    for data in tqdm(train_loader, total=len(train_loader)):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['labels'].to(device)
        #print(label)
        optimizer.zero_grad()
        output = model(input_ids = input_ids, attention_mask = attention_mask)
        loss = loss_fn(output.logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() 
    return total_loss / len(train_loader)


####
from sklearn.metrics import accuracy_score, classification_report

def validation_step(model, validation_loader, loss_fn, device):
    model.eval()
    predictions = []
    actual_labels = []
    valid_loss = []
    
    with torch.no_grad():
        for batch in validation_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # Assuming labels are one-hot encoded
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            valid_loss.append(loss.item())
            
            _, preds = torch.max(outputs.logits, dim=1)
            #print("pred : ", preds)
            predictions.extend(preds.cpu().tolist())
            
            # Convert one-hot encoded labels to class indices
            #actual_indices = torch.argmax(labels, dim=1)
            #print("actual indice : ", actual_indices)
            actual_labels.extend(labels.cpu().tolist())
            #print("actual label : ", actual_labels)
    
    accuracy = accuracy_score(actual_labels, predictions)
    #class_report = classification_report(actual_labels, predictions)
    
    return np.mean(valid_loss), accuracy

def save_checkpoint(model, checkpoint_filename):
    '''
        save the checkpoint after training
    '''

    state = {
        'classifier': model.classifier,
        'model_state_dict': model.state_dict(),
        'class_to_idx' : model.class_to_idx
    }

    torch.save(state, checkpoint_filename)

def main():
    #wandb.init(project = "bert_corana_sent_anal")
    dataset = MyDataset(csv_file = config["csv_fil"],
                        tokenizer_name= config["model_name"],
                        max_length= config["max_length"],
                        )
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, shuffle=True)
    
    train_loader = dataloader(dataset=train_dataset,
                              batch_size= config["batch_size"],
                              shuffle=True)
    
    valid_loader = dataloader(dataset=test_dataset, batch_size= config["batch_size"], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomModel(model_name = config["model_name"], n_classes=5)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr = config["learning_rate"])
    for epoch in range(config["n_epochs"]):
        #loss_train = train_step(model, train_loader, optimizer, loss_fn, device)
        loss_valid, accuracy = validation_step(model = model, validation_loader = valid_loader, device=device, loss_fn=loss_fn)
    
        #wandb.log({"loss_train":loss_train,
         #         "loss_valid" : loss_valid,
          #        "accuracy" : accuracy})


    #sauvegarder notre model
    #save_checkpoint(model, "checkpoints.pt")
if __name__ == "__main__":
    main()
