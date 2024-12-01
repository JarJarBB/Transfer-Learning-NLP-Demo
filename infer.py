from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

model_path = "./saved_model"
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)


def translate(input_text: str) -> str:
    inputs = loaded_tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = loaded_model.generate(
        **inputs, max_length=40, num_beams=4, early_stopping=True
    )
    translated_text = loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text


english_sentence = "He couldn’t afford a taxi after the concert, so he braced himself to ride Shanks' mare through the darkened streets."
french_translation = translate(english_sentence)

print(f"English: {english_sentence}")
print(f"French: {french_translation}")
