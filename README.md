# LLMs-Financial-Sentiment-Analysis

This repository showcases **financial sentiment classification** using **Large Language Models (LLMs)** and classic transformer models. It focuses on **fine-tuning** and **comparing** multiple architectures—ranging from **BERT**, **RoBERTa**, to **LLaMA (LoRA-based)**—on a 3-class sentiment task (negative, neutral, positive). The project also demonstrates **deployment** on [Hugging Face Spaces](https://huggingface.co/spaces/ChienChung/llama-lora-sentiment) for real-time inference.

## Table of Contents
1. [Overview](#overview)
2. [Models & Results](#models--results)
3. [Data Augmentation](#data-augmentation-ongoing)
4. [Project Structure](#project-structure)
5. [How to Run](#how-to-run)
6. [Future Work](#future-work)
7. [License](#license)

---

## Overview

**Objective:** Predict whether a financial news headline (or short text) is **negative**, **neutral**, or **positive**. Such sentiment analysis can be applied to:
- **Market research** and trading signals
- **News analytics** and risk assessment
- **Investor relations** and user feedback

**Key Highlights:**
- **Fine-tuning LLaMA with LoRA** for domain-specific sentiment.
- Baseline vs. Fine-tuned performance comparison.
- Additional experiments with **BERT** and **RoBERTa**.
- Deployment on Hugging Face Spaces (interactive Gradio demo).
- **Data augmentation** (synonym replacement, back-translation).

This project is designed so that everyone can get a sense of the end-to-end ML pipeline, from data loading to final inference.

---

## Models & Results

Experimented with four primary approaches:

### 1. **LLaMA (LoRA)**
- **Baseline** (no fine-tuning):
  - **Accuracy:** ~43.7%  
  - **F1:** ~38.7%  
  - <details>
    <summary>Baseline LLaMA Confusion Matrix</summary>

    ```bash
    [[ 12  47   1]
     [ 78 199   5]
     [ 41 101   1]]
    ```
    </details>
    
- **Fine-tuned** (LoRA-based):
  - **Accuracy:** ~84.9%  
  - **F1:** ~85.0%  
  - Training used 4 epochs, `learning_rate=2e-5`, `weight_decay=0.01`, `per_device_train_batch_size=1` on Apple M2 Mac.
  - <details>
    <summary>Fine-tuned LLaMA (LoRA) Confusion Matrix</summary>

    ```bash
    [[ 48   9   3]
    [  5 243  34]
    [  2  20 121]]
    ```
    </details>


### 2. **BERT (base-uncased)**
- **Fine-tuned**:
  - **Accuracy:** ~80.4%  
  - **F1:** ~80.7%  
  - `eval_loss` ~0.74  
  - Confusion matrix suggests the model occasionally confuses neutral vs. positive.
  - <details>
    <summary>Fine-tuned BERT Confusion Matrix</summary>

    ```bash
    [[ 44   8   8]
     [ 10 223  49]
     [  4  16 123]]
    ```
    </details>

### 3. **RoBERTa**
- **Fine-tuned**:
  - **Accuracy:** ~82.4%  
  - **F1:** ~82.6%  
  - `eval_loss` ~0.74  
  - Slower inference time than BERT in these tests.
  - <details>
    <summary>Fine-tuned BERT Confusion Matrix</summary>

    ```bash
    [[ 46  11   3]
     [ 12 235  35]
     [  2  22 119]]
    ```
    </details>

### 4. **FinBERT** (Exploratory)
- Observed high baseline accuracy without additional fine-tuning in certain splits.  
- Fine-tuning can help in more specialized domains, but we saw minimal or mixed improvement.  
- This portion is optional in the final report, as domain-specific FinBERT may already be quite strong.

---

## Data Augmentation
We explored several data augmentation techniques to boost the minority (negative) class:
- **Synonym Replacement** (WordNet-based)
- **Back-Translation** (English → Chinese → English)
- *Random Swap, Headline-Style Simplification, etc.*
  
Preliminary results showed a ~2.7% accuracy gain for BERT primarily from Synonym Replacement and Back-Translation, while other methods yielded marginal improvements.

---

## Project Structure

| File/Folder                      | Description |
|----------------------------------|--------------------------------------------------|
| 📂 `data/`                         | Financial headlines with sentiment labels |
| 📂 `experiments`                 | Model experiments & evaluation |
| 📄 `Llama_LoRA.ipynb` | LLaMA + LoRA fine-tuning notebook |
| 📄 `llama3-lora-finetuned/`         | Final LoRA weights for LLaMA |
| 📄 `app.py`            | Gradio app for LLaMA + LoRA inference|
| 📄 `requirements.txt`            | Dependencies |
| 📄 `README.md`                   | This file |
| 📄 `.gitignore`                  | Ignore unnecessary files |

- **`data/`** contains the CSV dataset (3-class: negative, neutral, positive).
- **`experiments/`** houses separate notebooks for each model variant (BERT, RoBERTa, etc.).
- **`llama-lora-finetuned/`** holds the fine-tuned LoRA weights for LLaMA.
- **`app.py`** runs a Gradio interface, loading the base LLaMA + LoRA checkpoint for real-time predictions.

---

## How to Run
#### Online: [Hugging Face Spaces](https://huggingface.co/spaces/ChienChung/llama-lora-sentiment)
#### Local:
1. **Clone this repo**:
   ```bash
   git clone https://github.com/tim3959951/LLMs-Financial-Sentiment-Analysis.git
   cd LLMs-Financial-Sentiment-Analysis
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Export** your Hugging Face token (see **Note** below):
   ```bash
   export HF_TOKEN=your_token_here  # macOS/Linux
   set HF_TOKEN=your_token_here  # Windows
   ```
4. **Run local Gradio app (for LLaMA + LoRA)**:
   ```bash
   python app.py
   ```
   - By default, it launches at http://127.0.0.1:7860.
5. **Explore notebooks**:
   - Inside `experiments/BERT_ROBERTA_Finetuned.ipynb`, etc.

> **Note**: This project used a gated model, you need appropriate Hugging Face credentials. Make sure to [request access](https://huggingface.co/meta-llama/) and include your token.
> 
> To run the app successfully, you must:
> 1. **Request access to the model** here: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
> 2. **Log in and generate your access token**: https://huggingface.co/settings/tokens Create a token with `read` permissions.
> 3. **Export your token to the environment** before running the app:   
>
>   On Max/Linux:
>   ```bash
>   export HF_TOKEN=your_token_here
>   ```
>   On Windows:
>   ```bash
>   set HF_TOKEN=your_token_here
>   ```
---

## Future Work

- **Extended RAG (Retrieval-Augmented Generation):** Combine LLM inference with a knowledge base for improved context.
- **PEFT expansions:** Explore QLoRA or other parameter-efficient methods to reduce memory usage.
- **Data Augmentation** finalization: systematically apply to LLaMA to see if negative-class recall improves further.
- **Deployment**: Potentially push a Docker container or integrate CI/CD for streamlined model updates.

---

## License

This data is licensed under the [License](http://creativecommons.org/licenses/by-nc-sa/3.0/.). For the LLaMA base model, please consult [Meta’s model license](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and [Hugging Face Terms](https://huggingface.co/meta-llama).

---

**Thank you for visiting!**  
For questions or collaboration opportunities, feel free to open an issue or reach out via [LinkedIn](https://www.linkedin.com/in/tim-cch).
