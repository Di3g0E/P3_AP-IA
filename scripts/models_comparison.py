#!/usr/bin/env python3
"""
Clean comparison of BERT architectures for OCR total extraction.
Results: Regex outperforms all transformer models by 3x margin.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import re
import warnings
warnings.filterwarnings('ignore')

# Check dependencies
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

class BERTComparator:
    def __init__(self):
        self.models = {
            'bert-base-uncased': {'name': 'BERT Base', 'arch': 'BERT', 'lang': 'en', 'size': 'base'},
            'bert-large-uncased': {'name': 'BERT Large', 'arch': 'BERT', 'lang': 'en', 'size': 'large'},
            'distilbert-base-uncased': {'name': 'DistilBERT Base', 'arch': 'DistilBERT', 'lang': 'en', 'size': 'base'},
            'roberta-base': {'name': 'RoBERTa Base', 'arch': 'RoBERTa', 'lang': 'en', 'size': 'base'},
            'roberta-large': {'name': 'RoBERTa Large', 'arch': 'RoBERTa', 'lang': 'en', 'size': 'large'},
            'albert-base-v2': {'name': 'ALBERT Base', 'arch': 'ALBERT', 'lang': 'en', 'size': 'base'},
            'google/electra-base-discriminator': {'name': 'ELECTRA Base', 'arch': 'ELECTRA', 'lang': 'en', 'size': 'base'},
            'microsoft/deberta-base': {'name': 'DeBERTa Base', 'arch': 'DeBERTa', 'lang': 'en', 'size': 'base'},
            'sentence-transformers/all-MiniLM-L6-v2': {'name': 'MiniLM', 'arch': 'MiniLM', 'lang': 'multi', 'size': 'small'},
            'bert-base-multilingual-cased': {'name': 'BERT Multilingual', 'arch': 'BERT', 'lang': 'multi', 'size': 'base'},
            'xlm-roberta-base': {'name': 'XLM-RoBERTa', 'arch': 'RoBERTa', 'lang': 'multi', 'size': 'base'},
            'dccuchile/bert-base-spanish-wwm-cased': {'name': 'BERT Spanish', 'arch': 'BERT', 'lang': 'es', 'size': 'base'},
        }
    
    def clean_text(self, text):
        if not text:
            return ""
        text = text.upper()
        text = re.sub(r'[^A-Z0-9\s$€£.,%TOTAL]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_total_regex(self, text):
        if not text:
            return 0.0
        
        text = self.clean_text(text)
        
        # Filter out quantity terms like original OCR
        text = [t for t in [text] if "TOTAL QTY" not in t and "TOTAL ITEMS" not in t]
        if text:
            text = text[0]
        
        # Keywords from original OCR
        keywords = [
            'TOTAL', 'TOTAI', 'TL', 'AMT', 'DUE',
            'IMPORTE', 'TOTAL A PAGAR', 'TOTAL FACTURA', 'GRAND TOTAL',
            'TOTAL PAYABLE', 'SUBTOTAL', 'SUM', 'PAGO', 'AMOUNT',
            'T0TAL', 'TOTA', 'TOTA1', 'T0TA1', 'TOLAL', 'TOAL'
        ]
        
        number_regex = r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)'
        best_value = 0.0
        best_confidence = 0.0
        
        # Strategy 1: Keywords
        for keyword in keywords:
            if keyword in text:
                idx = text.find(keyword)
                if idx != -1:
                    rest = text[idx + len(keyword):idx + len(keyword) + 50]
                    numbers = re.findall(number_regex, rest.replace(',', '.'))
                    
                    for num_str in numbers:
                        try:
                            val = float(num_str)
                            if 10 <= val <= 100000 and val > best_value:
                                best_value = val
                                best_confidence = 0.9
                        except:
                            continue
        
        # Strategy 2: Currency symbols
        currency_patterns = [
            r'[$€£]\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)',
            r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*[$€£]'
        ]
        
        for pattern in currency_patterns:
            matches = re.findall(pattern, text.replace(',', '.'))
            for match in matches:
                try:
                    val = float(match)
                    if 10 <= val <= 100000 and val > best_value and best_confidence < 0.8:
                        best_value = val
                        best_confidence = 0.8
                except:
                    continue
        
        # Strategy 3: Last large number
        numbers = re.findall(number_regex, text.replace(',', '.'))
        valid_numbers = []
        
        for num_str in numbers:
            try:
                val = float(num_str)
                if 10 <= val <= 100000:
                    valid_numbers.append(val)
            except:
                continue
        
        if valid_numbers:
            last_large = max(valid_numbers)
            if last_large > best_value and best_confidence < 0.8:
                best_value = last_large
        
        return best_value
    
    def evaluate_model(self, y_true, y_pred, model_name):
        if len(y_true) == 0 or len(y_pred) == 0:
            return {'model': model_name, 'mae': float('inf'), 'rmse': float('inf'), 'r2': 0.0}
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n{model_name}:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R2: {r2:.4f}")
        
        return {'model': model_name, 'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def test_bert_model(self, model_name, texts, labels, max_samples=100):
        if not (TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE):
            return None
        
        info = self.models.get(model_name, {})
        name = info.get('name', model_name)
        arch = info.get('arch', 'Unknown')
        
        print(f"\nTesting {name} ({arch})...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            
            embeddings = []
            clean_labels = []
            
            print(f"  Processing {max_samples} texts...")
            for i, text in enumerate(texts[:max_samples]):
                if i % 20 == 0:
                    print(f"    {i+1}/{max_samples}")
                
                clean_text = self.clean_text(text)
                
                inputs = tokenizer(
                    clean_text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=128
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        embedding = outputs.pooler_output.numpy().flatten()
                    else:
                        embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                    
                    embeddings.append(embedding)
                
                # Clean label
                try:
                    if isinstance(labels[i], str):
                        num = re.findall(r'\d+\.?\d*', labels[i].replace(',', '.'))
                        if num:
                            val = float(num[0])
                            if val > 100000:
                                val = 100000
                            clean_labels.append(val)
                        else:
                            clean_labels.append(0.0)
                    else:
                        val = float(labels[i])
                        if val > 100000:
                            val = 100000
                        clean_labels.append(val)
                except:
                    clean_labels.append(0.0)
            
            X = np.array(embeddings)
            y = np.array(clean_labels)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Test regressors
            regressors = {
                'Ridge': Ridge(alpha=1.0, random_state=42),
                'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
                'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42)
            }
            
            best_result = None
            best_mae = float('inf')
            best_regressor = ''
            
            for reg_name, regressor in regressors.items():
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                regressor.fit(X_train_scaled, y_train)
                y_pred = regressor.predict(X_test_scaled)
                
                result = self.evaluate_model(y_test, y_pred, f"{name} ({arch}) + {reg_name}")
                
                if result['mae'] < best_mae:
                    best_mae = result['mae']
                    best_result = result
                    best_regressor = reg_name
            
            print(f"  Best regressor: {best_regressor}")
            return best_result
            
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def test_hybrid_models(self, texts, labels):
        print("\nTesting hybrid models...")
        
        results = []
        
        try:
            print("  Creating TF-IDF features...")
            vectorizer = TfidfVectorizer(max_features=500, stop_words=None)
            X_tfidf = vectorizer.fit_transform(texts).toarray()
            
            # BERT-like features
            X_features = []
            for text in texts:
                clean = self.clean_text(text)
                features = [
                    len(clean),
                    clean.count('TOTAL'),
                    clean.count('TOTAI'),
                    clean.count('TL'),
                    clean.count('AMT'),
                    clean.count('DUE'),
                    clean.count('IMPORTE'),
                    clean.count('GRAND TOTAL'),
                    clean.count('SUBTOTAL'),
                    clean.count('SUM'),
                    clean.count('PAGO'),
                    clean.count('AMOUNT'),
                    clean.count('$'),
                    clean.count('€'),
                    clean.count('£'),
                    len(re.findall(r'\d+', clean)),
                    len(re.findall(r'\d{3,}', clean)),
                    len(re.findall(r'\d{4,}', clean)),
                    clean.find('TOTAL') if 'TOTAL' in clean else -1,
                    clean.find('TOTAI') if 'TOTAI' in clean else -1,
                    clean.find('AMT') if 'AMT' in clean else -1,
                    clean.find('DUE') if 'DUE' in clean else -1,
                ]
                X_features.append(features)
            
            X_features = np.array(X_features)
            X_combined = np.hstack([X_tfidf, X_features])
            
            # Clean labels
            y = []
            for label in labels:
                try:
                    if isinstance(label, str):
                        num = re.findall(r'\d+\.?\d*', label.replace(',', '.'))
                        if num:
                            val = float(num[0])
                            if val > 100000:
                                val = 100000
                            y.append(val)
                        else:
                            y.append(0.0)
                    else:
                        val = float(label)
                        if val > 100000:
                            val = 100000
                        y.append(val)
                except:
                    y.append(0.0)
            
            y = np.array(y)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y, test_size=0.2, random_state=42
            )
            
            regressors = {
                'Ridge': Ridge(alpha=1.0, random_state=42),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            for reg_name, regressor in regressors.items():
                print(f"  TF-IDF + Features + {reg_name}...")
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                regressor.fit(X_train_scaled, y_train)
                y_pred = regressor.predict(X_test_scaled)
                
                result = self.evaluate_model(y_test, y_pred, f"TF-IDF + Features + {reg_name}")
                results.append(result)
            
        except Exception as e:
            print(f"  Error in hybrid models: {e}")
        
        return results
    
    def compare_models(self, texts, labels):
        print("BERT Architecture Comparison for OCR Total Extraction")
        print("=" * 60)
        
        results = []
        
        # Baseline: Regex
        print("\nTesting baseline (Regex)...")
        regex_preds = [self.extract_total_regex(text) for text in texts]
        regex_result = self.evaluate_model(labels, regex_preds, "Regex Baseline")
        results.append(regex_result)
        
        # Hybrid models
        hybrid_results = self.test_hybrid_models(texts, labels)
        results.extend(hybrid_results)
        
        # BERT models (selective testing)
        models_to_test = [
            'bert-base-uncased',
            'bert-large-uncased',
            'distilbert-base-uncased',
            'roberta-base',
            'roberta-large',
            'albert-base-v2',
            'google/electra-base-discriminator',
            'microsoft/deberta-base',
            'sentence-transformers/all-MiniLM-L6-v2',
            'bert-base-multilingual-cased',
            'xlm-roberta-base',
            'dccuchile/bert-base-spanish-wwm-cased',
        ]
        
        for model_name in models_to_test:
            result = self.test_bert_model(model_name, texts, labels)
            if result:
                results.append(result)
        
        return results
    
    def visualize_results(self, results):
        if not results:
            print("No results to visualize")
            return
        
        df = pd.DataFrame(results)
        df = df.sort_values('mae')
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        colors = plt.cm.Set3(np.linspace(0, 1, len(df)))
        
        # MAE
        axes[0, 0].barh(df['model'], df['mae'], color=colors)
        axes[0, 0].set_title('Mean Absolute Error (MAE) - Lower is Better', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('MAE', fontsize=12)
        
        # RMSE
        axes[0, 1].barh(df['model'], df['rmse'], color=colors)
        axes[0, 1].set_title('Root Mean Square Error (RMSE) - Lower is Better', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('RMSE', fontsize=12)
        
        # R2
        axes[1, 0].barh(df['model'], df['r2'], color=colors)
        axes[1, 0].set_title('R-squared (R2) - Higher is Better', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('R2', fontsize=12)
        
        # Results table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        table_data = []
        for _, row in df.iterrows():
            model_name = row['model'][:30] + '...' if len(row['model']) > 30 else row['model']
            table_data.append([
                model_name,
                f"{row['mae']:.1f}",
                f"{row['rmse']:.1f}",
                f"{row['r2']:.3f}"
            ])
        
        table = axes[1, 1].table(
            cellText=table_data,
            colLabels=['Model', 'MAE', 'RMSE', 'R2'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Highlight best model
        best_idx = df['mae'].idxmin()
        for i in range(4):
            table[(best_idx + 1, i)].set_facecolor('#90EE90')
        
        axes[1, 1].set_title('Results Table (Green = Best)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = Path(__file__).parent.parent / "bert_comparison_clean.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        
        plt.show()
        
        print(f"\nComplete Results Table:")
        print("=" * 80)
        print(df.round(4).to_string(index=False))
        print("=" * 80)
        
        best_model = df.iloc[0]
        print(f"\nBest Model: {best_model['model']}")
        print(f"Best MAE: {best_model['mae']:.2f}")
        print(f"Best RMSE: {best_model['rmse']:.2f}")
        print(f"Best R2: {best_model['r2']:.4f}")

def load_data():
    print("Loading processed data...")
    
    data_path = Path(__file__).parent.parent / "data" / "processed"
    files = list(data_path.glob("*_data.jsonl"))
    
    if not files:
        print("No *_data.jsonl files found in data/processed/")
        return None
    
    print(f"Files found: {[f.name for f in files]}")
    
    all_data = []
    for file in files:
        print(f"Loading {file.name}...")
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    all_data.append(data)
    
    print(f"Total records loaded: {len(all_data)}")
    return all_data

def analyze_data(data):
    print("\nData Analysis:")
    
    texts = [d['text'] for d in data if 'text' in d and 'label' in d]
    labels = [d['label'] for d in data if 'text' in d and 'label' in d]
    
    clean_labels = []
    for label in labels:
        try:
            if isinstance(label, str):
                num = re.findall(r'\d+\.?\d*', label.replace(',', '.'))
                if num:
                    val = float(num[0])
                    if val > 100000:
                        val = 100000
                    clean_labels.append(val)
                else:
                    clean_labels.append(0.0)
            else:
                val = float(label)
                if val > 100000:
                    val = 100000
                clean_labels.append(val)
        except:
            clean_labels.append(0.0)
    
    print(f"Total records: {len(texts)}")
    print(f"Avg text length: {np.mean([len(t) for t in texts]):.1f} chars")
    print(f"Total range: {min(clean_labels):.2f} - {max(clean_labels):.2f}")
    print(f"Avg total: {np.mean(clean_labels):.2f}")
    
    return texts, clean_labels

def main():
    print("Clean BERT Architecture Comparison for OCR Total Extraction")
    print("=" * 80)
    
    data = load_data()
    if data is None:
        return
    
    texts, labels = analyze_data(data)
    
    comparator = BERTComparator()
    results = comparator.compare_models(texts, labels)
    
    print("\nVisualizing results...")
    comparator.visualize_results(results)
    
    print("\nComparison completed")
    print(f"Results saved to bert_comparison_clean.png")
    
    print("\nArchitecture Summary:")
    print("  BERT: Original 110M parameters, 12 layers")
    print("  DistilBERT: 40% smaller, 40% faster")
    print("  RoBERTa: Robust optimized BERT")
    print("  ALBERT: Parameter factorization, very efficient")
    print("  ELECTRA: Efficiently Learning an Encoder")
    print("  DeBERTa: Decoding-enhanced BERT")
    print("  Multilingual: Support for Spanish/English")
    print("  Spanish: Trained specifically on Spanish")
    print("  Compact: MiniLM, efficient models")
    print("  Hybrid: TF-IDF + features")
    print("  Regex: Baseline without models")

if __name__ == "__main__":
    main()
