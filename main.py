
from data.datafile import THUMOS
from data.utils import get_data_loader
from training import train_one_epoch
from lib.modeling import make_meta_arch
from lib.utils import make_optimizer, make_scheduler

from config.config import load_config, load_default_config, _update_config, _merge 

if __name__ == "__main__":


    cfgfile = "config/thumos_i3d.yaml"
    cfg = load_config(cfgfile)
    cfg = _update_config(cfg)

    dataset_cfg = cfg['dataset']
         
    dataset = THUMOS(split='train', feat_stride=dataset_cfg['feat_stride'], num_frames=dataset_cfg['num_frames'])

    dataloader = get_data_loader(dataset, batch_size=1,  num_workers=1)

    model = make_meta_arch(cfg['model_name'],**cfg['model'])
    # print(model)
    optimizer = make_optimizer(model, cfg['opt'])

    num_iters_per_epoch = len(dataloader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    model.train()
    
    for epoch_no in range(cfg['opt']['epochs']):
        train_one_epoch(dataloader, model, optimizer, scheduler, epoch_no)
    
