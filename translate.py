from pathlib import Path
from config import get_config, get_weights_file_path 
from model import build_transformer
from tokenizers import Tokenizer
from datasets import load_dataset
from datasett import BilingualDataset, casual_mask_func
import torch
import sys
from torch.utils.data import DataLoader ,Dataset,random_split
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
def translate_2( sentence ):
    config=get_config()
    max_seq_len=config['seq_len']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_trg = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size(), config["seq_len"], config['seq_len'], d_model=config['d_model']).to(device)

    # Load the pretrained weights
    model_filename = get_weights_file_path(config,config["preload"])
    #print(f'Loading model from {model_filename}')
    state = torch.load(model_filename,weights_only=False)
    model.load_state_dict(state['model_state_dict'])

    model.eval()
    count=0
    console_width=80
    ds_val=BilingualDataset(sentence,tokenizer_src,tokenizer_trg,config["lang_src"],config["lang_trg"],config["seq_len"],mode=True)
    val_dataloader=DataLoader(ds_val,batch_size=1,shuffle=True)
    with torch.no_grad():
        for batch in val_dataloader:
            
            count+=1

            print('-'*console_width)
            encoder_input=batch["encoder_input"].to(device)
            encoder_mask=batch["encoder_mask"].to(device)
            assert encoder_input.size(0)==1
            model_output=greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_trg,max_seq_len,device)

            model_out_text=tokenizer_trg.decode(model_output.detach().cpu().numpy())

            print('-'*console_width)

            print(f"predicted: {model_out_text}")
            if count==1:
                break

string1="Die Politiker diskutieren über neue Gesetze." #example 1
string2="Die Wahlen sind nächstes Jahr" #example 2
string3="Es ist eine Schande, daß es uns nicht gelingt, den Rassismus und die Fremdenfeindlichkeit in der Europäischen Union auszumerzen, die in meinem eigenen Land ebenso weit verbreitet sind wie in der gesamten EU."
#read sentence from argument
string4="Die Umweltpolitik spielt eine immer wichtigere Rolle, da der Klimawandel eine der größten Herausforderungen unserer Zeit darstellt und nur durch internationale Zusammenarbeit gelöst werden kann"
#We believe, however, that the Commission's strategic plan needs to be debated within a proper procedural framework, not only on the basis of an oral statement here in the European Parliament, but also on the basis of a document which is adopted in the Commission and which describes this programme over the five-year period."
string5="Die Regierung plant, in den nächsten Jahren mehr in Bildung und Gesundheit zu investieren, um sicherzustellen, dass alle Bürger Zugang zu den besten Möglichkeiten haben und gleichzeitig die wirtschaftliche Stabilität des Landes zu gewährleisten."
string6="Angesichts der fortschreitenden technologischen Entwicklungen und der dringenden Notwendigkeit, den Klimawandel zu bekämpfen, steht Deutschland vor der komplexen Aufgabe, seine industrielle Produktion zu modernisieren, den Übergang zu erneuerbaren Energien zu beschleunigen und gleichzeitig sicherzustellen, dass die soziale Gerechtigkeit und der Wohlstand der Bevölkerung nicht gefährdet werden, während es sich in einem zunehmend globalisierten Wettbewerb behauptet."
translation_examples=[string1,string2,string3,string4,string5,string6]
for i,example in enumerate(translation_examples):
    print(f"example {i+1}")
    print(translate_2(example))
