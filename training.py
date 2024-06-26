import torch
import torch.nn as nn
import torch.utils.data
from lib.modeling import make_meta_arch
from lib.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)

import wandb

def train_one_epoch(dataloader, model, optimizer, scheduler, epoch_no):
    
    print(f"Training epoch {epoch_no}")

    for i, data in enumerate(dataloader, 0):

        #print(data)
        optimizer.zero_grad()
        #print(data)
        loss = model(data)
        print(loss)

        loss['final_loss'].backward()

        optimizer.step()
        scheduler.step()

        wandb.log(loss)

        
    


