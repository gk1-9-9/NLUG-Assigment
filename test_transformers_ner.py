"""
Test script for Hugging Face Transformers Named Entity Recognition
This script demonstrates NER using pre-trained BERT model from Hugging Face
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def test_transformers_ner():
    """Test Hugging Face Transformers NER on sample text"""
    print("=" * 60)
    print("Testing Hugging Face Transformers NER")
    print("=" * 60)
    
    # Load model and tokenizer
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    
    print(f"\nLoading model: {model_name}")
    print("This may take a few minutes on first run...")
    
    try:
        # Create NER pipeline
        ner_pipeline = pipeline(
            "ner",
            model=model_name,
            tokenizer=model_name,
            aggregation_strategy="simple"  # Group sub-tokens
        )
        print("✓ Successfully loaded model and tokenizer")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Sample texts for testing
    sample_texts = [
        "Tesla opened a new Gigafactory in Austin, Texas, investing over $5 billion in the facility.",
        "Narendra Modi visited Japan in May 2023 to attend the G7 summit held in Hiroshima.",
        "Amazon, led by Andy Jassy, is expanding its logistics operations across Southeast Asia.",
        "The International Monetary Fund, based in Washington D.C., revised its global growth forecast for 2024."
    ]
    
    print("\n" + "=" * 60)
    print("NER Results on Sample Texts")
    print("=" * 60)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n--- Text {i} ---")
        print(f"Input: {text}")
        print("\nEntities:")
        
        # Get NER predictions
        entities = ner_pipeline(text)
        
        if entities:
            for ent in entities:
                print(f"  - {ent['word']:20} | {ent['entity_group']:10} | Score: {ent['score']:.4f}")
        else:
            print("  No entities found")
    
    # Entity label mapping for CoNLL-2003
    print("\n" + "=" * 60)
    print("Entity Labels (CoNLL-2003)")
    print("=" * 60)
    entity_labels = {
        "PER": "Person",
        "ORG": "Organization",
        "LOC": "Location",
        "MISC": "Miscellaneous"
    }
    
    for label, description in entity_labels.items():
        print(f"  {label:10} : {description}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_transformers_ner()
