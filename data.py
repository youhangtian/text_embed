import torch 
from torch.utils.data import Dataset, DataLoader 


class DataCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer 
        self.max_length = max_length 

    def _ids_tensor(self, texts):
        text_ids = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        return torch.Tensor(text_ids)

    def __call__(self, records):
        texts = [record['text'] for record in records]
        text_ids = self._ids_tensor(texts)

        texts_pos = [record['text_pos'] for record in records]
        text_pos_ids = self._ids_tensor(texts_pos)

        if 'text_neg' in records[0].keys():
            texts_neg = [record['text_neg'] for record in records]
            text_neg_ids = self._ids_tensor(texts_neg)
            return {
                'text_ids': text_ids,
                'text_pos_ids': text_pos_ids,
                'text_neg_ids': text_neg_ids,
            }
        elif 'label' in records[0].keys():
            labels = [record['label'] for record in records]
            labels = torch.tensor(labels, dtype=torch.float32)
            return {
                'text_ids': text_ids,
                'text_pos_ids': text_pos_ids,
                'labels': labels,
            }
        else:
            return {
                'text_ids': text_ids,
                'text_pos_ids': text_pos_ids,
            }


def create_dataloader(dataset, 
                      tokenizer, 
                      batch_size,
                      max_length=512,
                      num_workers=0,
                      drop_last=False,
                      shuffle=False):
    data_collator = DataCollator(tokenizer, max_length)
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=drop_last, 
    )
    return data_loader 

def get_dataloader(dataset, tokenizer, batch_size=256):
    train_dataset = dataset['train']
    if 'dev' in dataset:
        val_dataset = dataset['dev']
    elif 'validation' in dataset:
        val_dataset = dataset['validation']
    else:
        val_dataset = None 

    train_dl = create_dataloader(
        train_dataset, 
        tokenizer,
        batch_size,
        shuffle=True, 
        drop_last=True
    )
    val_dl = create_dataloader(
        val_dataset, 
        tokenizer,
        batch_size,
        shuffle=False, 
        drop_last=False
    ) if val_dataset else None 

    return train_dl, val_dl 
