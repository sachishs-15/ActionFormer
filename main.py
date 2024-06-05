
from data.datafile import THUMOS
from data.utils import get_data_loader
from training import train_one_epoch
from lib.modeling import make_meta_arch
from lib.utils import make_optimizer, make_scheduler

cfg = {

"model": 
{
  "fpn_type": "identity",
  "max_buffer_len_factor": 6.0,
  "n_mha_win_size": 19,
}, 

"opt": 
{
  "learning_rate": 0.0001,
  "epochs": 1,
  "weight_decay": 0.05
}

}

if __name__ == "__main__":

    dataset = THUMOS()
    dataloader = get_data_loader(dataset, batch_size=1,  num_workers=1)

    model = make_meta_arch(cfg['model'])
    optimizer = make_optimizer(model, cfg['opt'])

    num_iters_per_epoch = len(dataloader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    for epoch_no in range(cfg['opt']['epochs']):
        train_one_epoch(dataloader, model, optimizer, scheduler, epoch_no)
    
