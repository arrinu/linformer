import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
from pathlib import Path
import spacy
import re
import pandas as pd

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    text = text.lower() #Lowercase
    text = re.sub(r'@[\w_-]+', '', text) #Removes @user
    text = re.sub(r'[^\w\s]', '', text) #Removes special Chars
    doc = nlp(text) #Tokenizes
    tokens = [token.lemma_ for token in doc if not token.is_stop] #Removes Stop words and lemmatizes
    processed_text = ' '.join(tokens) #Joins it back into a string    
    return processed_text

def preprocess_ds(df):
    df['text'] = df['text'].apply(preprocess_text)
    return df

class DSCreator(Dataset):
    
    def __init__(self, ds, tokenizer, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        pair = self.ds.iloc[index]
        src_text = pair['text']
        src_label = pair['label']
        
        input_tokens = self.tokenizer.encode(src_text).ids
        
        num_padding_token = self.seq_len - len(input_tokens) - 2
        
        if num_padding_token<0:
            raise ValueError('Input sentence exceeds Sequence Length limit')
        
        enc_inp = torch.cat(
            [
                self.sos_token,
                torch.tensor(input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_padding_token, dtype=torch.int64),
            ],
            dim=0,
        )
        
        label = torch.tensor(src_label, dtype=torch.int64)
        
        assert enc_inp.size(0) == self.seq_len
        
        return{
            "encoder_input": enc_inp,
            "encoder_mask": (enc_inp != self.pad_token).unsqueeze(0).unsqueeze(0).float(), 
            "label": label,
        }

def get_sentences():
    df = pd.read_parquet("dataset/train.parquet")
    df = preprocess_ds(df)
    for sentence in df["text"]:
        yield sentence
 
def build_tokenizer(config):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_sentences(), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def create_ds(config):
    train_data_raw = pd.read_parquet("dataset/train.parquet")
    val_data_raw = pd.read_parquet("dataset/validation.parquet")
    train_pp = preprocess_ds(train_data_raw)
    val_pp = preprocess_ds(val_data_raw)
    tokenizer = build_tokenizer(config)

    train_ds = DSCreator(train_pp, tokenizer, config['seq_len'])
    val_ds = DSCreator(val_pp, tokenizer, config['seq_len'])
    
    max_len = 0
    for item in train_pp['text']:
        src_id = tokenizer.encode(item).ids
        max_len = max(max_len, len(src_id))
    print(f"Max Length for sentence = {max_len}")
        
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle = True)
    
    return train_dataloader, val_dataloader, tokenizer

def create_testds(config):
    test_data_raw = pd.read_parquet("dataset/test.parquet")
    test_pp = preprocess_ds(test_data_raw)
    
    tokenizer_path = Path(config['tokenizer_file'])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    test_ds = DSCreator(test_pp, tokenizer, config['seq_len'])
    
    test_dataloader = DataLoader(test_ds, batch_size = 1, shuffle= True)
    
    return test_dataloader, tokenizer