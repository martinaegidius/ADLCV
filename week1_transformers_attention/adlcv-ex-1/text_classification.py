import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
from torchtext import data, datasets, vocab
import tqdm
from transformer import TransformerClassifier, to_device
import json
import random
import string


NUM_CLS = 2
VOCAB_SIZE = 50_000
SAMPLED_RATIO = 0.2
MAX_SEQ_LEN = 512

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data_iter(sampled_ratio=0.2, batch_size=16):
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
    # Reduce dataset size
    reduced_tdata, _ = tdata.split(split_ratio=sampled_ratio)
    # Create train and test splits
    train, test = reduced_tdata.split(split_ratio=0.8)
    print('training: ', len(train), 'test: ', len(test))
    TEXT.build_vocab(train, max_size= VOCAB_SIZE - 2)
    LABEL.build_vocab(train)
    train_iter, test_iter = data.BucketIterator.splits((train, test), 
                                                       batch_size=batch_size, 
                                                       device=to_device()
    )

    return train_iter, test_iter


def main(embed_dim=128, num_heads=4, num_layers=4, num_epochs=20,
         pos_enc='fixed', pool='max', dropout=0.0, fc_dim=None,
         batch_size=16, lr=1e-4, warmup_steps=625, 
         weight_decay=1e-4, gradient_clipping=1
    ):

    
    
    loss_function = nn.CrossEntropyLoss()

    train_iter, test_iter = prepare_data_iter(sampled_ratio=SAMPLED_RATIO, 
                                            batch_size=batch_size
    )
    print("Training batch-size: ",train_iter.batch_size)

    
    model = TransformerClassifier(embed_dim=embed_dim, 
                                  num_heads=num_heads, 
                                  num_layers=num_layers,
                                  pos_enc=pos_enc,
                                  pool=pool,  
                                  dropout=dropout,
                                  fc_dim=fc_dim,
                                  max_seq_len=MAX_SEQ_LEN, 
                                  num_tokens=VOCAB_SIZE, 
                                  num_classes=NUM_CLS,
                                  )
    
    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    epoch_train_loss = []
    epoch_val_acc = []
    # training loop
    try:
        for e in range(num_epochs):
            print(f'\n epoch {e}')
            model.train()
            batch_loss = 0.0
            for batch in tqdm.tqdm(train_iter):
                opt.zero_grad()
                #print(f"batch: {batch}")
                input_seq = batch.text[0]
                batch_size, seq_len = input_seq.size()
                label = batch.label - 1
                if seq_len > MAX_SEQ_LEN:
                    input_seq = input_seq[:, :MAX_SEQ_LEN]
                #print(f"input_seq shape: {input_seq.shape}")
                out = model(input_seq)
                #print(f"output shape: {out.shape}")
                loss = loss_function(out,label)
                batch_loss += loss
                #print("Loss becomes: ", loss)
                loss.backward() # backward
                # if the total gradient vector has a length > 1, we clip it back down to 1.
                if gradient_clipping > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                opt.step()
                sch.step()
            epoch_train_loss.append(batch_loss/(4000.0/train_iter.batch_size)) #4000 samples

            with torch.no_grad():
                model.eval()
                tot, cor= 0.0, 0.0
                for batch in test_iter:
                    input_seq = batch.text[0]
                    batch_size, seq_len = input_seq.size()
                    label = batch.label - 1
                    if seq_len > MAX_SEQ_LEN:
                        input_seq = input_seq[:, :MAX_SEQ_LEN]
                    out = model(input_seq).argmax(dim=1)
                    tot += float(input_seq.size(0))
                    cor += float((label == out).sum().item())
                acc = cor / tot
                print(f'-- {"validation"} accuracy {acc:.3}')
                epoch_val_acc.append(acc)
    except KeyboardInterrupt:
        break
    return epoch_train_loss, epoch_val_acc

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    #print("Random string of length", length, "is:", result_str)
    return result_str

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    param_cfg = {"embed_dim":128, "num_heads":4, "num_layers":4, "num_epochs":20,
         "pos_enc":'fixed', "pool":'max', "dropout":0.0, "fc_dim":None,
         "batch_size":16, "lr":1e-4, "warmup_steps":625, 
         "weight_decay":1e-4, "gradient_clipping":1}
    exp_name = get_random_string(5) + ".json"
    print("Running experiment ",exp_name)
    #train
    train_loss,val_acc = main(**param_cfg)
    #save results to dict and write to disk
    result_cfg = {"params":param_cfg,"epoch loss":train_loss,"accuracy":val_acc}
    with open(exp_name,"w") as f:
        json.dump(result_cfg,f)

    print("Succesfully wrote json. Exit")
    


