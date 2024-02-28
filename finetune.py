import torch 
from tqdm import tqdm 
from transformers import get_cosine_schedule_with_warmup 
from sentence_transformers import SentenceTransformer 
from datasets import load_from_disk 
from accelerate import Accelerator 
from accelerate.utils import set_seed 

from data import get_dataloader
from model import TorchModel
from eval.eval import eval_st_model 
from utils import get_config_from_yaml, get_logger

cfg = get_config_from_yaml('cfg.yaml')
logger = get_logger(cfg.output_dir)
logger.info(cfg)

model0 = SentenceTransformer(cfg.st_model)
model0[1].pooling_mode_cls_token = True
model0[1].pooling_mode_mean_tokens = False 

dataset = load_from_disk(cfg.ft_dataset)
train_dl, val_dl = get_dataloader(dataset, model0.tokenizer, batch_size=cfg.batch_size)

torch_model = TorchModel(model0)

epochs = cfg.epochs

lr = 8e-6
num_training_params = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)
if num_training_params <= 80_000_000: lr = 1e-4
elif num_training_params <= 200_000_000: lr = 5e-5

weight_decay = cfg.weight_decay
no_decay_keywords = ('bias', 'LayerNorm', 'layernorm')

parameters = list(torch_model.named_parameters())
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in parameters if not any(nd in n for nd in no_decay_keywords)],
        'weight_decay': weight_decay,
    },
    {
        'params': [p for n, p in parameters if any(nd in n for nd in no_decay_keywords)],
        'weight_decay': 0.0, 
    }
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer, 
    num_warmup_steps=0,
    num_training_steps=int(len(train_dl) * epochs), 
)

accelerator = Accelerator(
    mixed_precision='no',
    gradient_accumulation_steps=1,
    log_with=None,
)
accelerator.init_trackers(cfg.output_dir)

set_seed(cfg.seed)
train_dl = accelerator.prepare(train_dl)
val_dl = accelerator.prepare(val_dl) if val_dl is not None else None 
model = accelerator.prepare(torch_model)
optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
logger.info(f'accelerator prepared!')

steps = 0
for epoch in range(epochs):
    if cfg.mteb_zh_eval:
        torch_model.model.eval()
        eval_res = eval_st_model(torch_model.model, logger)
        logger.info(f'epoch{epoch} eval res: {eval_res:.6f}')

    torch_model.train()
    for batch in tqdm(train_dl):
        with accelerator.accumulate(torch_model):
            optimizer.zero_grad()
            output = torch_model(batch)
            loss = output['loss']
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

        if steps % cfg.log_freq == 0:
            logger.info(f'[epoch{epoch}][steps{steps}] loss: {loss.item():.6f}')
        steps += 1

if cfg.mteb_zh_eval:
    torch_model.model.eval()
    eval_res = eval_st_model(torch_model.model, logger)
    logger.info(f'final eval res: {eval_res:.6f}')

accelerator.wait_for_everyone()
logger.info('Training finished')

torch_model.model.save(f'{cfg.output_dir}/{cfg.output_dir}')
logger.info(f'Model saved in {cfg.output_dir}/{cfg.output_dir}')
