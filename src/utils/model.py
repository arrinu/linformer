import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module): 
    def __init__(self, d_model, intermediate_size, dropout):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model,intermediate_size)
        self.linear2 = nn.Linear(intermediate_size,d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class LinformerAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout, seq_len, k):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model (Input Vector Embedding dims) not divisible by h(Number of Heads)"
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.proj_k = nn.Linear(seq_len, k, bias=False)
        self.proj_v= nn.Linear(seq_len, k, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def linformer_attention(query, key, value, mask, proj_k, proj_v, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        key = key.transpose(-1,-2) # (b, h, s, d) -> (b, h, d, s)
        value = value.transpose(-1,-2) # (b, h, s, d) -> (b, h, d, s)
        
        key = proj_k(key) # (b, h, d, s) -> (b, h, d, k)
        mask = proj_k(mask) # (b, h, d, d) -> (b, h, d, k)
        value = proj_v(value) # (b, h, d, s) -> (b, h, d, k)
        
        key = key.transpose(-1,-2) # (b, h, d, k) -> (b, h, k, d)
        value = value.transpose(-1,-2) # (b, h, d, k) -> (b, h, k, d)
        
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # (b, h, s, d) * (b, h, d, k)  -> (b, h, s, k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            
        attention_scores = attention_scores.softmax(dim=-1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores # (b, h, s, k) * (b, h, k, d)  -> (b, h, s, d), 

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v) 

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = LinformerAttentionBlock.linformer_attention(query, key, value, mask, self.proj_k, self.proj_v, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        return self.w_o(x)

class ResidualConnection(nn.Module):
        def __init__(self, features, dropout):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
    
class ClassificationHead(nn.Module):
    def __init__(self, d_model, op_label_size):
        super().__init__()
        self.proj = nn.Linear(d_model, op_label_size)
    
    def forward(self,x):
        x = torch.mean(x, dim=1) # (b, s, d) -> (b, d)
        return torch.log_softmax(self.proj(x), dim =-1) # (b, d) -> (b, n)
    
class EncoderBlock(nn.Module):
    def __init__(self, features,  selfAttentionHead, feedforward, dropout):
        super().__init__()
        self.selfAttentionHead = selfAttentionHead

        self.feedforward = feedforward
        self.skip = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.skip[0](x, lambda x: self.selfAttentionHead(x, x, x, src_mask))
        x = self.skip[1](x, self.feedforward)
        return x
    
class Encoder(nn.Module):
    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class Linformer_cls(nn.Module):
    def __init__(self, encoder, src_embed, src_pos, classification_head):
        super().__init__()
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.encoder = encoder
        self.classification_head = classification_head
        
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def project(self, x):
        return self.classification_head(x)
    
def build_model(src_vocab_size, label_vocab_size, src_seq_len, reduced_len, d_model, N, h, dropout, intermediate_size):
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = LinformerAttentionBlock(d_model,h,dropout, src_seq_len, reduced_len)
        feed_forward = FeedForwardBlock(d_model, intermediate_size, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)
        
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
        
    classification_head = ClassificationHead(d_model, label_vocab_size)
        
    model = Linformer_cls(encoder, src_embed, src_pos,  classification_head)
        
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
        
    return model