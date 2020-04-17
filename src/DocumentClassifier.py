from transformers import BertTokenizer, BertForSequenceClassification
import torch

class DocumentClassifier:
    def __init__(self, weight="bert-base-uncased"):
        self.weight = weight
    
        self.tokenizer = BertTokenizer.from_pretrained(self.weight)
        self.model = BertForSequenceClassification.from_pretrained(self.weight)

    def predict(self, target):
        input_ids = torch.tensor(self.tokenizer.encode(target,
                                 add_special_tokens=True)).unsqueeze(0)
        labels = torch.tensor([1]).unsqueeze(0)
        outputs = model(input_ids, labels=labels)
        return outputs[2]