from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Peed911/Roberta_corona_class")
model = AutoModelForSequenceClassification.from_pretrained("Peed911/Roberta_corona_class")
label_mapping = {
    "Extremely Negative": 0,
    "Negative" : 1,
    "Neutral": 2,
    "Positive": 3,
    "Extremely Positive" : 4,
}
labels_name = [ "Extremely Negative", "Negative", "Neutral", "Positive", "Extremely Positive"]
# Define the prediction function
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_names = labels_name[predicted_class]
    return predicted_class,  predicted_names

app = FastAPI()

class Item(BaseModel):
    text : str 

@app.post("/predict")
def read_root(text : Item):
    pred_class, pred_name = classify_text(str(text))
    return {"predicted classe" : pred_class, "predicted name" : pred_name }
    #return {"pred" : str(text)}
