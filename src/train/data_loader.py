import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import os
from config import TEXT_EMBEDDING_MODEL

class DementiaDataset(Dataset):
    def __init__(
        self,
        audio_csv_path,
        embedding_csv_path,
        text_dir,
        label_value=None,
        tokenizer_model=TEXT_EMBEDDING_MODEL,
        max_length=512
    ):
        self.audio_data = pd.read_csv(audio_csv_path)
        self.embedding_data = pd.read_csv(embedding_csv_path)
        """
        self.audio_data = self.audio_data.replace([np.inf, -np.inf], np.nan)
        self.embedding_data = self.embedding_data.replace([np.inf, -np.inf], np.nan)
        for col in self.audio_data.columns:
            if col not in ('patient_id', 'class'):
                self.audio_data[col] = pd.to_numeric(self.audio_data[col], errors='coerce')
        for col in self.embedding_data.columns:
            if col != 'patient_id':
                self.embedding_data[col] = pd.to_numeric(self.embedding_data[col], errors='coerce')
        self.audio_data = self.audio_data.fillna(0.0)
        self.embedding_data = self.embedding_data.fillna(0.0)
        """
        self.text_dir = text_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_length = max_length
        self.data = pd.merge(
            self.audio_data,
            self.embedding_data,
            on='patient_id',
            suffixes=('_audio', '_embedding')
        )

        self.label_value = label_value
        self.data['label'] = self.label_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        patient_id = data_row['patient_id']
        text_file_path = os.path.join(self.text_dir, f'{patient_id}.txt')

        with open(text_file_path, 'r') as file:
            text = file.read()

        text_tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Squeeze out the batch dimension added by return_tensors='pt'
        text_tokens = {key: value.squeeze(0) for key, value in text_tokens.items()}
        
        audio_feature_columns = [
            col for col in self.audio_data.columns if col not in ('patient_id', 'class')
        ]
        embedding_feature_columns = [
            col for col in self.embedding_data.columns if col != 'patient_id'
        ]
        audio_features = data_row[audio_feature_columns].values.astype(float)
        embedding_features = data_row[embedding_feature_columns].values.astype(float)
        label = torch.tensor(data_row['label'], dtype=torch.float32)

        audio_tensor = torch.tensor(audio_features, dtype=torch.float32)
        embedding_tensor = torch.tensor(embedding_features, dtype=torch.float32)

        return text_tokens, audio_tensor, embedding_tensor, label


class TestDistDataset(Dataset):
    def __init__(
        self,
        audio_csv_path,
        embedding_csv_path,
        text_dir,
        tokenizer_model=TEXT_EMBEDDING_MODEL,
        max_length=512,
    ):
        self.audio_data = pd.read_csv(audio_csv_path)
        self.embedding_data = pd.read_csv(embedding_csv_path)
        self.audio_data = self.audio_data.replace([np.inf, -np.inf], np.nan)
        self.embedding_data = self.embedding_data.replace([np.inf, -np.inf], np.nan)
        for col in self.audio_data.columns:
            if col not in ('patient_id', 'class'):
                self.audio_data[col] = pd.to_numeric(self.audio_data[col], errors='coerce')
        for col in self.embedding_data.columns:
            if col != 'patient_id':
                self.embedding_data[col] = pd.to_numeric(self.embedding_data[col], errors='coerce')
        self.audio_data = self.audio_data.fillna(0.0)
        self.embedding_data = self.embedding_data.fillna(0.0)

        self.text_dir = text_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_length = max_length

        self.data = pd.merge(self.audio_data, self.embedding_data, on='patient_id')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_row = self.data.iloc[idx]
        patient_id = data_row['patient_id']
        text_file_path = os.path.join(self.text_dir, f'{patient_id}.txt')
        with open(text_file_path, 'r') as file:
            text = file.read()

        text_tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        # Squeeze out the batch dimension added by return_tensors='pt'
        text_tokens = {key: value.squeeze(0) for key, value in text_tokens.items()}

        audio_feature_columns = [col for col in self.audio_data.columns if col not in ('patient_id', 'class')]
        embedding_feature_columns = [col for col in self.embedding_data.columns if col != 'patient_id']
        audio_features = data_row[audio_feature_columns].values.astype(float)
        embedding_features = data_row[embedding_feature_columns].values.astype(float)

        audio_tensor = torch.tensor(audio_features, dtype=torch.float32)
        embedding_tensor = torch.tensor(embedding_features, dtype=torch.float32)

        return text_tokens, audio_tensor, embedding_tensor, str(patient_id)

def create_full_dataset(
    ad_text_dir,
    cn_text_dir,
    ad_csv,
    cn_csv,
    ad_embedding_csv,
    cn_embedding_csv
):
    ad_dataset = DementiaDataset(
        ad_csv,
        ad_embedding_csv,
        ad_text_dir,
        label_value=1
    )
    cn_dataset = DementiaDataset(
        cn_csv,
        cn_embedding_csv,
        cn_text_dir,
        label_value=0
    )
    full_dataset = ConcatDataset([ad_dataset, cn_dataset])
    return full_dataset


def create_testdist_dataset(testdist_dir, audio_csv, embedding_csv):
    return TestDistDataset(
        audio_csv_path=audio_csv,
        embedding_csv_path=embedding_csv,
        text_dir=testdist_dir,
    )


def create_test_dataset(test_dir, audio_csv, embedding_csv, labels_csv, tokenizer_model=TEXT_EMBEDDING_MODEL):
    """Create test dataset with ground truth labels from task1.csv"""
    # Read labels
    labels_df = pd.read_csv(labels_csv)
    labels_dict = {}
    for _, row in labels_df.iterrows():
        patient_id = row['ID']
        label = 1 if row['Dx'] == 'ProbableAD' else 0
        labels_dict[patient_id] = label
    
    # Read audio and embedding data
    audio_data = pd.read_csv(audio_csv)
    embedding_data = pd.read_csv(embedding_csv)
    
    # Merge on patient_id
    data = pd.merge(audio_data, embedding_data, on='patient_id', suffixes=('_audio', '_embedding'))
    
    # Add labels
    data['label'] = data['patient_id'].map(labels_dict)
    
    # Create dataset class instance
    class TestDatasetWithLabels(Dataset):
        def __init__(self, data, audio_data, embedding_data, text_dir, tokenizer_model, max_length=512):
            self.data = data
            self.audio_data = audio_data
            self.embedding_data = embedding_data
            self.text_dir = text_dir
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
            self.max_length = max_length
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            data_row = self.data.iloc[idx]
            patient_id = data_row['patient_id']
            text_file_path = os.path.join(self.text_dir, f'{patient_id}.txt')
            
            with open(text_file_path, 'r') as file:
                text = file.read()
            
            text_tokens = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Squeeze out the batch dimension added by return_tensors='pt'
            text_tokens = {key: value.squeeze(0) for key, value in text_tokens.items()}
            
            audio_feature_columns = [col for col in self.audio_data.columns if col not in ('patient_id', 'class')]
            embedding_feature_columns = [col for col in self.embedding_data.columns if col != 'patient_id']
            audio_features = data_row[audio_feature_columns].values.astype(float)
            embedding_features = data_row[embedding_feature_columns].values.astype(float)
            label = torch.tensor(data_row['label'], dtype=torch.float32)
            
            audio_tensor = torch.tensor(audio_features, dtype=torch.float32)
            embedding_tensor = torch.tensor(embedding_features, dtype=torch.float32)
            
            return text_tokens, audio_tensor, embedding_tensor, label
    
    return TestDatasetWithLabels(data, audio_data, embedding_data, test_dir, tokenizer_model)
