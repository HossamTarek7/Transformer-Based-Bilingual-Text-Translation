import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_trg,src_lang,trg_lang,seq_len,mode=False):
        super().__init__()
        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_trg=tokenizer_trg
        self.src_lang=src_lang
        self.trg_lang=trg_lang
        self.seq_len = seq_len
        self.mode=mode
        
        if "[SOS]" not in tokenizer_trg.get_vocab():
            print("[SOS] token is missing from the vocabulary.")

        self.sos_token = torch.tensor([tokenizer_trg.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):        return len(self.ds)
    def __getitem__(self,idx):
        if self.mode:
            
            src_text=self.ds

            encoded_src=self.tokenizer_src.encode(src_text).ids
            #raise error if neg

            encoded_num_pads=self.seq_len-len(encoded_src)-2
            if len(encoded_src) > self.seq_len - 2:
                encoded_src = encoded_src[:self.seq_len - 2]
           
            src_tensor=torch.cat([
                self.sos_token,
                torch.tensor(encoded_src,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*encoded_num_pads,dtype=torch.int64)
            ],dim=0)

            assert src_tensor.size(0)==self.seq_len

            return {
                "encoder_input":src_tensor,
                "encoder_mask":(src_tensor!=self.pad_token).unsqueeze(0).unsqueeze(0).int(),
                "src_text":src_text
            }

        else:
            
            src_text=src_traget_pair['translation'][self.src_lang]
            src_traget_pair=self.ds[idx]
            trg_text=src_traget_pair['translation'][self.trg_lang]
            src_text=src_traget_pair['translation'][self.src_lang]
            trg_text=src_traget_pair['translation'][self.trg_lang]

            encoded_src=self.tokenizer_src.encode(src_text).ids
            decoded_trg=self.tokenizer_trg.encode(trg_text).ids
            #raise error if neg

            encoded_num_pads=self.seq_len-len(encoded_src)-2
            dencoded_num_pads=self.seq_len-len(decoded_trg)-1
            if len(encoded_src) > self.seq_len - 2:
                encoded_src = encoded_src[:self.seq_len - 2]
            if len(decoded_trg) > self.seq_len - 1:
                decoded_trg = decoded_trg[:self.seq_len - 1]

            src_tensor=torch.cat([
                self.sos_token,
                torch.tensor(encoded_src,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*encoded_num_pads,dtype=torch.int64)
            ],dim=0)
            trg_tensor=torch.cat([
                self.sos_token,
                torch.tensor(decoded_trg,dtype=torch.int64),
                torch.tensor([self.pad_token]*dencoded_num_pads,dtype=torch.int64)
            ],dim=0)
            label= torch.cat([
                torch.tensor(decoded_trg,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dencoded_num_pads,dtype=torch.int64)
            ],dim=0)
            assert src_tensor.size(0)==self.seq_len
            assert trg_tensor.size(0)==self.seq_len
            assert label.size(0)==self.seq_len
            return {
                "encoder_input":src_tensor,
                "decoder_input":trg_tensor,
                "encoder_mask":(src_tensor!=self.pad_token).unsqueeze(0).unsqueeze(0).int(),
                "decoder_mask":(trg_tensor!=self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask_func(trg_tensor.size(0)),
                "label":label,
                "src_text":src_text,
                "trg_text":trg_text
            }

def casual_mask_func(size):
    mask=torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask== 0
        