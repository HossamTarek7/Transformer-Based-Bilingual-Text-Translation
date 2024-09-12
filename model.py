
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader ,Dataset,random_split
from pathlib import Path
import numpy as np
import spacy
import random
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard



class InputEmbedding(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embeddings=nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embeddings(x) * np.sqrt(self.d_model)

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len,dropout)->None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        self.postions=torch.arange(0, seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(self.postions * div_term)
        pe[:,1::2] = torch.cos(self.postions * div_term)
        pe=pe.unsqueeze(0)
        #pe shape (1,seq_len,d_model)
        self.register_buffer('pe',pe)
    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1]]).requires_grad_(False)
        return self.dropout(x)


# %%
class LayerNormalization(nn.Module):
    def __init__(self,eps=1e-6):
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))
        self.beta=nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.alpha*(x-mean)/(std+self.eps)+self.beta
    


# %%
class FeedForwardBlock(nn.Module):
    def __init__(self,d_model,dff,dropout):
        super().__init__()
        self.d_model=d_model
        self.dff=dff
        self.dropout=nn.Dropout(dropout)
        self.linear1=nn.Linear(d_model,dff)
        self.linear2=nn.Linear(dff,d_model)
    def forward(self,x):
        x=self.dropout(torch.relu(self.linear1(x)))
        x=self.linear2(x)
        return x
    

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,head,dropout):
        super().__init__()
        self.d_model = d_model
        self.head = head
        assert( d_model % head == 0),"Embedding size should be divisible by head"

        self.d_k=d_model//head
        self.W_out=nn.Linear(self.d_model,self.d_model,bias=False) # garb mara t7oto w mara tshelo
        self.values=nn.Linear(self.d_model,self.d_model,bias=False)
        self.keys=nn.Linear(self.d_model,self.d_model,bias=False)
        self.qureies=nn.Linear(self.d_model,self.d_model,bias=False)
        self.dropout=nn.Dropout(dropout)
    @staticmethod
    def attention(query,keys,values,mask,dropout):
        d_k=query.shape[-1]
        attemtion_scores=(query@keys.transpose(-2,-1))/np.sqrt(d_k)
        if mask is not None:
            attemtion_scores=attemtion_scores.masked_fill(mask==0,float("-1e9"))
        attentions=torch.softmax(attemtion_scores,dim=-1)
        if dropout is not None:
            attentions=dropout(attentions)
        out=attentions@values
        return out,attentions
    def forward(self,query,keys,values,mask):
        values=self.values(values)
        keys=self.keys(keys)
        query=self.qureies(query)
        values=values.view(values.shape[0],values.shape[1],self.head,self.d_k).transpose(1,2)
        keys=keys.view(keys.shape[0],keys.shape[1],self.head,self.d_k).transpose(1,2)
        query=query.view(query.shape[0],query.shape[1],self.head,self.d_k).transpose(1,2)
        x,self.attention_scores=MultiHeadAttention.attention(query,keys,values,mask,self.dropout)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.d_model)
        out=self.W_out(x)
        return out
    

# %%
class ResdiualConnection(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm=LayerNormalization()
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x))) #x+(1-self.dropout(sublayer(self.norm(x))))*(x))

# %% [markdown]
# try a different method for postional encoding

# %%
class Encoder_Block(nn.Module):
    def __init__(self,self_attention,ff_block,dropout):
        super().__init__()
        self.attention=self_attention
        self.ff_block=ff_block
        self.dropout=nn.Dropout(dropout)
        self.resdiual_connection=nn.ModuleList([ResdiualConnection(dropout) for _ in range(2)])
    def forward(self,x,src_mask):    
        x= self.resdiual_connection[0](x,lambda x:self.attention(x,x,x,src_mask))
        x= self.resdiual_connection[1](x,self.ff_block)
        return x
    

# %%
class Encoder(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
    def forward(self,x,src_mask):
        for layer in self.layers:
            x=layer(x,src_mask)
        return self.norm(x)

# %%
class DecoderBlock(nn.Module):
    def __init__(self, self_attention,cross_attention,feed_forward,dropout):
        super().__init__()
        self.self_attention=self_attention
        self.cross_attention=cross_attention
        self.feed_forward=feed_forward
        self.resdiual_connection=nn.ModuleList([ResdiualConnection(dropout) for _ in range(3)])
    def forward(self,x,encoder_out,src_mask,trg_mask):
        x= self.resdiual_connection[0](x,lambda x:self.self_attention(x,x,x,trg_mask))
        x= self.resdiual_connection[1](x,lambda x:self.cross_attention(x,encoder_out,encoder_out,src_mask))
        x= self.resdiual_connection[2](x,self.feed_forward)
        return x

# %%
class Decoder(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization()
    def forward(self,x,encoder_out,src_mask,trg_mask):
        for layer in self.layers:
            x=layer(x,encoder_out,src_mask,trg_mask)
        return self.norm(x)

# %%
class ProjectionLayer(nn.Module):
    def __init__(self,d_model,trg_vocab_size):
        super().__init__()
        self.projection=nn.Linear(d_model,trg_vocab_size)
    def forward(self,x):
        out=self.projection(x)
        return  torch.log_softmax(out,dim=-1)

# %%
class Transformer(nn.Module):
    def __init__(self,encoder,decoder,src_embed,trg_embed,src_postion,trg_position,projection):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.trg_embed=trg_embed
        self.src_postion=src_postion
        self.trg_position=trg_position
        self.projection=projection
    def encode(self,src,src_mask):
        src=self.src_embed(src)
        src=self.src_postion(src)
        return self.encoder(src,src_mask)
    def decode(self,encoder_out,src_mask,trg,trg_mask):
        trg=self.trg_embed(trg)
        trg=self.trg_position(trg)
        return self.decoder(trg,encoder_out,src_mask,trg_mask)
    def project(self,x):
        return self.projection(x)
        
     
   

# %% [markdown]
# can be used for mutlible task but the same structure

# %%
def build_transformer(src_vocab_size,trg_vocab_size,src_seq_len,trg_seq_len,d_model=512,N=6,h=8,dropout=0.1,d_ff=2048)->Transformer:
    src_embed=InputEmbedding(d_model,src_vocab_size)  
    trg_embed=InputEmbedding(d_model,trg_vocab_size)
    
    src_position=PositionalEncoding(d_model,src_seq_len,dropout)
    trg_position=PositionalEncoding(d_model,trg_seq_len,dropout)
    
    encoder_blocks=[]
    for _ in range(N):
        encoder_self_attention=MultiHeadAttention(d_model,h,dropout)
        feed_forward=FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block=Encoder_Block(encoder_self_attention,feed_forward,dropout)
        encoder_blocks.append(encoder_block)
    decoder_blocks=[]
    for _ in range(N):
        Decoder_self_attention=MultiHeadAttention(d_model,h,dropout)
        decoder_cross_attention=MultiHeadAttention(d_model,h,dropout)
        feed_forward=FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block=DecoderBlock(Decoder_self_attention,decoder_cross_attention,feed_forward,dropout)
        decoder_blocks.append(decoder_block)
    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))
    projection_layer=ProjectionLayer(d_model,trg_vocab_size)
    transformer=Transformer(encoder,decoder,src_embed,trg_embed,src_position,trg_position,projection_layer)
    # intialize weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
