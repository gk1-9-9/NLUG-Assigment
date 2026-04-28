"""
Test script for spaCy Named Entity Recognition
This script demonstrates basic NER using spaCy
"""

import spacy

def test_spacy_ner():
    """Test spaCy NER on sample text"""
    print("=" * 60)
    print("Testing spaCy Named Entity Recognition")
    print("=" * 60)
    
    # Load English tokenizer, tagger, parser and NER
    try:
        nlp = spacy.load("en_core_web_sm")
        print("\n✓ Successfully loaded spaCy model 'en_core_web_sm'")
    except OSError:
        print("\n✗ Model not found. Please install it with:")
        print("  python -m spacy download en_core_web_sm")
        return
    
    # Sample text for testing
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
        
        # Process the text
        doc = nlp(text)
        
        # Extract and print entities
        if doc.ents:
            for ent in doc.ents:
                print(f"  - {ent.text:20} | {ent.label_:10} | {spacy.explain(ent.label_)}")
        else:
            print("  No entities found")
    
    # Test entity categories
    print("\n" + "=" * 60)
    print("Entity Categories in spaCy")
    print("=" * 60)
    entity_labels = {
        "PERSON": "People, including fictional",
        "ORG": "Companies, agencies, institutions",
        "GPE": "Countries, cities, states",
        "LOC": "Non-GPE locations, mountain ranges, bodies of water",
        "DATE": "Absolute or relative dates or periods",
        "MONEY": "Monetary values, including unit",
        "CARDINAL": "Numerals that do not fall under another type"
    }
    
    for label, description in entity_labels.items():
        print(f"  {label:10} : {description}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_spacy_ner()
