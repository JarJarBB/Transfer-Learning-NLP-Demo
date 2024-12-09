# Transfer Learning: NLP Demo

This demo shows how to fine-tune a translation model. In this case, the NLP model is a deep neural network translating from English to French.

The expression "to ride Shanks' mare,” which means "to go by foot,” is very uncommon and unknown to most translation models, including the base model used in this demo. However, after the fine-tuning is completed, the AI model is capable of accurately translating the idiom.

### Install Modules

```bash
pip3 install -r requirements.txt
```

### Fine-Tune The Translation Model

```bash
python3 train.py
```

### Infer (Translate) Using The Model

```bash
python3 infer.py
```
