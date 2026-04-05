# ============================================
# AutoArchitect — BERT Problem Analyzer
# ============================================

import torch
import pickle
import os
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

class ProblemAnalyzer:
    def __init__(self):
        print("Loading BERT analyzer...")
        base    = os.path.dirname(os.path.dirname(__file__))
        bert_path = os.path.join(base, 'models', 'bert')
        le_path   = os.path.join(base, 'models',
                                 'label_encoder.pkl')

        # Load tokenizer + model
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_path)
        self.model     = BertForSequenceClassification\
            .from_pretrained(bert_path)
        self.model.eval()

        # Load label encoder
        with open(le_path, 'rb') as f:
            self.le = pickle.load(f)

        # Pipeline configs
        self.pipelines = {
            'image': {
                'type':        'Image Classification',
                'dataset':     'CIFAR-10',
                'description': 'Detects visual patterns in images',
                'input_size':  3072,
                'num_classes': 10
            },
            'medical': {
                'type':        'Medical Image Analysis',
                'dataset':     'CIFAR-10',
                'description': 'Analyzes medical scans for diagnosis',
                'input_size':  3072,
                'num_classes': 10
            },
            'text': {
                'type':        'Text Classification',
                'dataset':     'CIFAR-10',
                'description': 'Understands language patterns',
                'input_size':  3072,
                'num_classes': 10
            },
            'security': {
                'type':        'Threat Detection',
                'dataset':     'CIFAR-10',
                'description': 'Detects anomalies and threats',
                'input_size':  3072,
                'num_classes': 10
            }
        }
        print("✅ BERT analyzer ready!")

    def analyze(self, problem_description):
        # Tokenize input
        encoding = self.tokenizer(
            problem_description,
            max_length     = 64,
            padding        = 'max_length',
            truncation     = True,
            return_tensors = 'pt'
        )

        # Get BERT prediction
        with torch.no_grad():
            outputs    = self.model(
                input_ids      = encoding['input_ids'],
                attention_mask = encoding['attention_mask']
            )
            probs      = torch.softmax(outputs.logits, dim=1)
            pred_idx   = probs.argmax().item()
            pred_label = str(self.le.classes_[pred_idx])
            confidence = round(probs.max().item() * 100, 1)

        # Get pipeline config
        config = self.pipelines.get(
            pred_label, self.pipelines['image'])

        return {
            'problem':      problem_description,
            'category':     pred_label,
            'type':         config['type'],
            'dataset':      config['dataset'],
            'description':  config['description'],
            'input_size':   config['input_size'],
            'num_classes':  config['num_classes'],
            'confidence':   confidence,
            'certain':      confidence > 70,
            'all_scores':   {
                str(self.le.classes_[i]):
                    round(probs[0][i].item() * 100, 1)
                for i in range(len(self.le.classes_))
            }
        }
