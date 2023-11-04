from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "distilbert-base-uncased"
text = "I just had a really nice dinner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(text, return_tensors="pt")

id2label = {0: "POSITIVE", 1: "NEGATIVE"}
label2id = {"POSITIVE": 0, "NEGATIVE": 1}
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model.forward(**inputs)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
outputs = classifier(text)

predicted_label_index = outputs.logits.argmax(-1).item()
predicted_label = id2label[predicted_label_index]
print(f"The predicted label for the text is: {predicted_label}")