import spacy
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from textblob import TextBlob

# Load the spaCy model for traditional NLP tasks
nlp = spacy.load("en_core_web_sm")

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_text(prompt, max_length=100):
    """Generate text using GPT-2 model."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text using the GPT-2 model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def analyze_text(text):
    """Analyze text using spaCy for named entities and part-of-speech tagging."""
    doc = nlp(text)

    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Extract part-of-speech tags
    pos_tags = [(token.text, token.pos_) for token in doc]

    return entities, pos_tags

def sentiment_analysis(text):
    """Perform sentiment analysis using TextBlob."""
    blob = TextBlob(text)
    return blob.sentiment

def main():
    # Example prompt for text generation
    prompt = "In the future, artificial intelligence will"

    # Generate text
    generated_text = generate_text(prompt)
    print("Generated Text:")
    print(generated_text)

    # Analyze the generated text
    print("\nAnalyzing Generated Text:")
    entities, pos_tags = analyze_text(generated_text)
    
    print("Named Entities:")
    for entity in entities:
        print(f"{entity[0]} ({entity[1]})")

    print("\nPart-of-Speech Tags:")
    for pos in pos_tags:
        print(f"{pos[0]} - {pos[1]}")

    # Perform sentiment analysis
    sentiment = sentiment_analysis(generated_text)
    print("\nSentiment Analysis:")
    print(f"Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")

if __name__ == "__main__":
    main()