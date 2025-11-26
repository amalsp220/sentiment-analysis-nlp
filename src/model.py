import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

class BERTSentimentClassifier(nn.Module):
    """BERT-based sentiment analysis model"""
    
    def __init__(self, num_classes=3, dropout=0.3, pretrained='bert-base-uncased'):
        super(BERTSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        logits = self.fc(dropout_output)
        return logits

class RoBERTaSentimentClassifier(nn.Module):
    """RoBERTa-based sentiment analysis model"""
    
    def __init__(self, num_classes=3, dropout=0.3):
        super(RoBERTaSentimentClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        logits = self.fc(dropout_output)
        return logits

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def preprocess(self, texts):
        encoding = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoding['input_ids'], encoding['attention_mask']
