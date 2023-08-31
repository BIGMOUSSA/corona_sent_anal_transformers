import gradio as gr
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

# Create a Gradio interface
iface = gr.Interface(
    fn=classify_text,
    inputs=gr.inputs.Textbox(),
    outputs=[gr.outputs.Label(num_top_classes=5), gr.outputs.Textbox()],  # Adjust the number of top classes as needed
)

# Launch the Gradio interface
iface.launch()
