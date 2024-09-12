# %%
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
from model import build_transformer
from config import get_config,get_weights_file_path
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from tqdm import tqdm
from datasets import load_dataset
from datasett import BilingualDataset, casual_mask_func
import numpy as np
import random
import warnings
# %%


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# %%
def get_all_sentences(ds,lang):
    for example in ds:
        yield example['translation'][lang]
def greedy_decode(model,soruce,sorce_mask,tokenizer_src,tokenizer_trg,max_seq_len ,device):
    sos_idx=tokenizer_src.token_to_id("[SOS]")
    eos_idx=tokenizer_trg.token_to_id("[EOS]")
    encoder_output=model.encode(soruce,sorce_mask)
    decoder_input=torch.empty(1,1).fill_(sos_idx).type_as(soruce).to(device)
    while True:
        if decoder_input.size(1)>=max_seq_len:
            break
        decoder_mask=casual_mask_func(decoder_input.size(1)).type_as(soruce).to(device)
        out=model.decode(encoder_output,sorce_mask,decoder_input,decoder_mask)
        probs=model.project(out[:,-1])
        _,next_word=torch.max(probs,dim=1)
        decoder_input=torch.cat([decoder_input,torch.empty(1,1).type_as(soruce).fill_(next_word.item()).to(device)],dim=1)
        if next_word.item()==eos_idx:
            break
    return decoder_input.squeeze(0)
def run_validation(model, ds_val,tokenizer_src,tokenizer_trg,max_seq_len ,device,print_massage,global_state,writer,num_examples=2):
    model.eval()
    count=0
 
    console_width=80

    with torch.no_grad():
        for batch in ds_val:
            
            count+=1

            print_massage('-'*console_width)
            print_massage(f"batch {count}/{len(ds_val)}")
            encoder_input=batch["encoder_input"].to(device)
            encoder_mask=batch["encoder_mask"].to(device)
            assert encoder_input.size(0)==1
            model_output=greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_trg,max_seq_len,device)
            soruce=batch["src_text"][0]
            target=batch["trg_text"][0]
            model_out_text=tokenizer_trg.decode(model_output.detach().cpu().numpy())

            print_massage('-'*console_width)
            print_massage(f"source: {soruce}")
            print_massage(f"target: {target}")
            print_massage(f"predicted: {model_out_text}")
            if count==num_examples:
                break
# %%
def get_or_build_tokenizer(config,ds,lang):
    tokenizer_path=Path(config["tokenizer_file"].format(lang))
    if not Path(tokenizer_path).exists():
        tokenizer=Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer=Whitespace()
        trainer=WordLevelTrainer(special_tokens =["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# %%
def get_ds(config):
    ds_raw=ds = load_dataset("stas/wmt14-en-de-pre-processed", split="train")

    length_of_ds = 80000
    ds_raw = ds_raw.select(
        range(length_of_ds))
    print(f"Loaded dataset of size {len(ds_raw)}")

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_trg = get_or_build_tokenizer(config, ds_raw, config["lang_trg"])

    # Split into train and validation datasets
    train_data_size = int(0.9 * len(ds_raw))
    val_data_size = len(ds_raw) - train_data_size
    train_ds_raw = ds_raw.select(range(train_data_size))
    val_ds_raw = ds_raw.select(range(train_data_size, len(ds_raw)))
    print(f"loaded dataset of size {len(ds_raw)}")
    tokenizer_src=get_or_build_tokenizer(config,ds_raw,config["lang_src"])
    tokenizer_trg=get_or_build_tokenizer(config,ds_raw,config["lang_trg"])

    train_data_size=int(0.9*len(ds_raw))
    val_data_size=len(ds_raw)-train_data_size
    train_ds_raw,val_ds_raw=random_split(ds_raw,[train_data_size,val_data_size])

    train_ds=BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_trg,config["lang_src"],config["lang_trg"],config["seq_len"])
    val_ds=BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_trg,config["lang_src"],config["lang_trg"],config["seq_len"])
    train_dataloader=DataLoader(train_ds,batch_size=config["batch_size"],shuffle=True)
    val_dataloader=DataLoader(val_ds,batch_size=1,shuffle=True)
    return train_dataloader,val_dataloader , tokenizer_src, tokenizer_trg

# %%
def get_model(config,vocab_src_len,vocab_trg_len):
    model=build_transformer(
    vocab_src_len,
    vocab_trg_len,
    config["seq_len"],
    config["seq_len"],
    d_model=config["d_model"]
)
    return model
def train(config):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_dataloader,val_dataloader , tokenizer_src, tokenizer_trg = get_ds(config)
    model=get_model(config,tokenizer_src.get_vocab_size(),tokenizer_trg.get_vocab_size()).to(device)
    #tensorboard
    writer=SummaryWriter(config["experiment_name"])
    optimizer=optim.Adam(model.parameters(),lr=config["lr"],eps=1e-9)
    inital_epoch=0
    global_step=0

    if config["preload"]:
        model_file_name=get_weights_file_path(config,config["preload"])
        
        print(f"loading model {model_file_name}")
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

        state=torch.load(model_file_name,weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        inital_epoch=state["epoch"]+1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step=state["global_step"]
        print(f"loaded model from epoch {inital_epoch} successfully")
    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[PAD]"),label_smoothing=0.1).to(device)
    for epoch in range(inital_epoch,config["num_epochs"]):
        print(f"epoch {epoch:02d}/{config['num_epochs']}")
        batch_iter=tqdm(train_dataloader,desc=f'processing epoch {epoch:02d}')
        for batch in batch_iter:
            #print current epoch
            
            model.train()
            encoder_input=batch["encoder_input"].to(device)
            decoder_input=batch["decoder_input"].to(device)
            encoder_mask=batch["encoder_mask"].to(device)
            decoder_mask=batch["decoder_mask"].to(device)
            encoder_output=model.encode(encoder_input,encoder_mask)
            decoder_output=model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
            projection_output=model.project(decoder_output)
            label=batch["label"].to(device)
            loss=loss_fn(projection_output.view(-1,tokenizer_trg.get_vocab_size()),label.view(-1))
            batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})
            writer.add_scalar("train_loss",loss.item(),global_step=global_step)
            writer.flush()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  
            global_step+=1
        model_file_name=get_weights_file_path(config,f'{epoch:02d}')
        torch.save({
            'epoch':epoch,  
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'global_step':global_step
            
        },model_file_name)
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_trg, config['seq_len'], device, lambda msg: batch_iter.write(msg), global_step, writer)
if __name__=="__main__":
    config=get_config()
    train(config)
    

        

