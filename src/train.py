import torch
import torch.nn as nn
from pathlib import Path
from utils.model import build_model
from utils.data import create_ds
from tqdm import tqdm
import wandb
import argparse
import pickle

def get_model(config, inp_vocab_size):
    model = build_model(
        inp_vocab_size,
        config['num_labels'],
        config['seq_len'],
        config['reduced_len'],
        config['d_model'],
        config['num_layers'],
        config['num_heads'],
        config['dropout'],
        config['intermediate_size']
    )
    return model


def train_and_validate(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    Path('weights').mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer = create_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps= 1e-9)
    
    init_epoch =0
    global_step = 0
    best_acc = 0
    best_epoch = 0
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    #TRAIN LOOP
    wandb.define_metric("Train_Loss", step_metric="global_step")
    wandb.define_metric("global_step")
    
    for epoch in range(init_epoch, config['epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {(epoch+1):02d}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            label = batch['label'].to(device) # (b, 1)
            
            encoder_output = model.encode(encoder_input, encoder_mask) # (b, s, d)
            classification_output = model.project(encoder_output) # (b, n)

            loss = loss_fn(classification_output, label)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            if global_step%10==0:
                wandb.log({'Train_Loss': loss.item(), 'global_step': global_step})
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
            
        # VAL LOOP
        model.eval()
        correct_predictions = 0
        
        wandb.define_metric("Epoch")
        wandb.define_metric("Val_Accuracy", step_metric = "epoch")

        with torch.no_grad():
            for batchidx, batch in enumerate(val_dataloader):
                encoder_input = batch['encoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                label = batch['label'].to(device)

                encoder_output = model.encode(encoder_input, encoder_mask)
                classification_output = model.project(encoder_output)
                model_output = torch.argmax(classification_output, dim=-1)

                correct_predictions += torch.sum(model_output.int() == label.int()).item()

        curr_accuracy = correct_predictions / len(val_dataloader)
        wandb.log({'Val_Accuracy': curr_accuracy, "Epoch": epoch})
        print(f'Validation Accuracy: {curr_accuracy}')
        if best_acc < curr_accuracy:
            print('Weights Updated!')
            best_acc = curr_accuracy
            model_filename = str(Path('.') / 'weights' / f"epoch_{epoch+1:02d}.pt")
            config['best_weight_path'] = model_filename
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, model_filename)
            
        config_file = 'config.pkl'
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--dmodel', type=int, default=256)
    parser.add_argument('--numlayers', type=int, default=8)
    parser.add_argument('--numheads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()
    
    config ={
        "batch_size": args.batchsize,
        "epochs": args.epoch, 
        "lr": args.lr,
        "d_model": args.dmodel,
        "seq_len": 256,
        "reduced_len": 128,
        "num_layers": args.numlayers,
        "num_heads": args.numheads,
        "dropout" : args.dropout,
        "intermediate_size" : 512,
        "num_labels": 2,
        "tokenizer_file" : "vocab.json",
        'best_weight_path' : None
    }
    
    wandb.login(key=args.key)
    wandb.init(
        project="Linformer Classification",
        config = config
    )
    
    train_and_validate(config)