import torch
import gradio as gr
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

hf_token = os.environ.get("HF_TOKEN")
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"  
lora_weights_path = "./llama3-lora-finetuned"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    token=hf_token,
    trust_remote_code=True
)

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    token=hf_token,
    trust_remote_code=True,
    num_labels=3
)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, lora_weights_path)


device = torch.device("cpu")
model.to(device)

label_names = ["negative", "neutral", "positive"]

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred_label = outputs.logits.argmax(dim=-1).item()
    return label_names[pred_label]

demo = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs="text",
    title="LLaMA LoRA Sentiment Demo"
)

if __name__ == "__main__":
    demo.launch()