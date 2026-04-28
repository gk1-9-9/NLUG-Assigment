from transformers import pipeline
import torch

class NERModel:
    def __init__(self, model_path="dbmdz/bert-large-cased-finetuned-conll03-english"):
        """Initialize NER model"""
        self.pipeline = pipeline(
            "ner",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )

    def extract_entities(self, text, min_score=0.8):
        """Extract entities from text, filtered by confidence threshold"""
        entities = self.pipeline(text)
        return [
            {
                "text": ent["word"],
                "label": ent["entity_group"],
                "score": round(float(ent["score"]), 4)
            }
            for ent in entities
            if ent["score"] >= min_score
        ]

    def extract_entities_batch(self, texts, batch_size=8, min_score=0.8):
        """Extract entities from multiple texts using native pipeline batching"""
        raw_results = self.pipeline(texts, batch_size=batch_size)
        return [
            [
                {
                    "text": ent["word"],
                    "label": ent["entity_group"],
                    "score": round(float(ent["score"]), 4)
                }
                for ent in result
                if ent["score"] >= min_score
            ]
            for result in raw_results
        ]

# Example usage
if __name__ == "__main__":
    ner_model = NERModel()
    text = "Tesla CEO Elon Musk announced a new Gigafactory in Austin, Texas."
    entities = ner_model.extract_entities(text)
    print(f"Text: {text}")
    print("Entities:", entities)
