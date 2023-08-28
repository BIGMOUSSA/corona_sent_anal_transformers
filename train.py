from torch.utils.data  import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn

config = {
    "model_name" : "bert-base-uncased",
    "max_length" : 80,
    "hidden_state" : 768
}
class MyDataset(Dataset):
    def __init__(self, csv_file, tokenizer_name, max_length):
        self.df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.df["text"][index]
        label = self.df["labels"][index]
        inputs = self.tokenizer(
            text = text,
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_tensors = "pt"
        )

        return {
            "input_ids" : inputs["input_ids"],
            "attention_mask" : inputs["attention_mask"],
            "labels" : torch.tensor(label)
        }

def dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset = dataset,
                      batch_size = batch_size,
                      shuffle = shuffle
                      )


class CustomModel(nn.Module):
    def __init__(self, model_name, n_classes):
        super(CustomModel, self).__init__()
        self.pretrained_model = BertModel.from_pretrained(model_name) #hidden_state 786 Bert_base
        self.classifier = nn.Linear(768, n_classes)
    
    def forward(self, input_ids, attention_mask):
        output = self.pretrained_model(input_ids = input_ids, attention_mask = attention_mask)
        output = self.classifier(output.last_hidden_state)
        return output



def train_step(model, train_loader, optimizer, loss_fn, device):
    model.train()
    
    total_loss  = 0

    for data in tqdm(train_loader, total=len(train_loader)):
        input_ids = data['input_ids'].squeeze(1).to(device)
        attention_mask = data['attention_mask'].to(device)
        label = data['label'].to(device)

        optimizer.zero_grad()
        output = model(input_ids = input_ids, attention_mask = attention_mask)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() 
    return total_loss / len(train_loader)


if __name__ == "__main__":
    csv_path = "data/cleaned/nlp_clean.csv"
    dataset = MyDataset(csv_file = csv_path,
                        tokenizer_name= config["model_name"],
                        max_length= config["max_length"],
                        )
    train_loader = dataloader(dataset=dataset,
                              batch_size=2,
                              shuffle=True)
    data = next(iter(train_loader))
    model = CustomModel(model_name = config["model_name"], n_classes=5)
    output = model(input_ids = data["input_ids"].squeeze(1), attention_mask = data["attention_mask"])
    print(output)