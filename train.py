import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

model_path = "./saved_model"
model_name = "Helsinki-NLP/opus-mt-en-fr"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
df = pd.read_json("data.json")
data = Dataset.from_pandas(df)


def tokenize_function(examples):
    model_inputs = tokenizer(examples["en"], truncation=True, padding="max_length")
    labels = tokenizer(examples["fr"], truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = data.map(tokenize_function, batched=True)
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    num_train_epochs=100,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="epoch",
)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)
trainer.train()
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print("model and tokenizer saved to", model_path)
