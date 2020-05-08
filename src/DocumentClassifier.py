from transformers import BertTokenizer, BertForSequenceClassification, BertJapaneseTokenizer
import torch
import torch.nn.functional as F

class DocumentClassifier:
    def __init__(self, weight="bert-base-uncased"):
        self.weight = weight
    
        self.tokenizer = BertTokenizer.from_pretrained(self.weight)
        self.model = BertForSequenceClassification.from_pretrained(self.weight)

    def predict(self, target):
        input_ids = torch.tensor(self.tokenizer.encode(target,
                                 add_special_tokens=True)).unsqueeze(0)
        labels = torch.tensor([1]).unsqueeze(0)
        outputs = self.model(input_ids, labels=labels)
        return F.softmax(outputs[1])

class DocumentClassifierForJapanese:
    def __init__(self, weight="bert-base-japanese-whole-word-masking"):
        self.weight = weight
    
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.weight)
        self.model = BertForSequenceClassification.from_pretrained(self.weight)

    def predict(self, target):
        input_ids = torch.tensor(self.tokenizer.encode(target,
                                 add_special_tokens=True)).unsqueeze(0)
        labels = torch.tensor([1]).unsqueeze(0)
        outputs = self.model(input_ids, labels=labels)
        return F.softmax(outputs[1])