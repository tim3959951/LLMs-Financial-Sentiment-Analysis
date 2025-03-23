import torch
import gradio as gr
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

hf_token = os.environ.get("HF_TOKEN")
base_model_name = "ChienChung/my-llama-1b"  
lora_weights_path = "./llama3-lora-finetuned"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    token=hf_token,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token 

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    token=hf_token,
    trust_remote_code=True,
    num_labels=3
)

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, lora_weights_path)


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device => {device}")
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
demo_description = """
    **This Space showcases a LoRA-fine-tuned LLaMA model for 3-class financial sentiment classification (negative/neutral/positive). Simply input a headline or short text related to finance, and the model will predict its sentiment**.
    
    **How to Use**:
    1. Enter text: Type or paste a financial news headline (or any short text) into the text box.
    2. Submit: Click the Submit button.
    3. View result: The predicted sentiment label—negative, neutral, or positive
    
    **Sample Questions**:
    1. Finnish Componenta has published its new long-term strategy for the period 2011-2015 with the aim of growing together with its customers.
    2. Affecto expects its net sales for the whole 2010 to increase from the 2009 level when they reached EUR 103 million.
    3. The company expects its net sales for the whole 2009 to remain below the 2008 level.
    4. BasWare 's product sales grew strongly in the financial period , by 24 percent.
    5. Fortum is looking to invest in several new production units , including a new waste-fired unit at its Brista combined heat and power ( CHP ) plant and a biofuels-based production unit at Vartan CHP plant.
    6. Sales in UK decreased by 10.5 % in January , while sales outside UK dropped by 17 %.
    """
demo = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs="text",
    title="LLaMA LoRA Financial News Headline Sentiment Demo",
    description=demo_description,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
