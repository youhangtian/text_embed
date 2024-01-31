import torch 
from torch.utils.data import Dataset, DataLoader 
from dataclasses import dataclass, fields 

class RecordType:
    Pair = 'pair'
    Triplet = 'triplet'
    Scored = 'scored' 

@dataclass(slots=True)
class PairRecord:
    text: str 
    text_pos: str 

@dataclass(slots=True)
class TripletRecord:
    text: str 
    text_pos: str 
    text_neg: str 

@dataclass(slots=True)
class ScoredRecord:
    sentence1: str 
    sentence2: str 
    label: float 

record_cls_map = {
    RecordType.Pair: PairRecord, 
    RecordType.Triplet: TripletRecord, 
    RecordType.Scored: ScoredRecord,
}

def get_type(record):
    for type, cls in record_cls_map.items():
        names = [field.name for field in fields(cls)]
        if all(name in record for name in names):
            return type 
    raise ValueError(f'Unknown record type: record: {record}')


class DataCollator:
    def __init__(self, tokenizer, max_length, record_type):
        self.tokenizer = tokenizer 
        self.max_length = max_length 
        self.record_type = record_type 

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
        if self.record_type == RecordType.Pair:
            texts = [record.text for record in records]
            text_ids = self._ids_tensor(texts)
            
            texts_pos = [record.text_pos for record in records]
            text_pos_ids = self._ids_tensor(texts_pos)

            return {
                'text_ids': text_ids,
                'text_pos_ids': text_pos_ids,
            }
        elif self.record_type == RecordType.Triplet:
            texts = [record.text for record in records]
            text_ids = self._ids_tensor(texts)

            texts_pos = [record.text_pos for record in records]
            text_pos_ids = self._ids_tensor(texts_pos)

            texts_neg = [record.text_neg for record in records]
            text_neg_ids = self._ids_tensor(texts_neg)

            return {
                'text_ids': text_ids,
                'text_pos_ids': text_pos_ids,
                'text_neg_ids': text_neg_ids,
            }
        elif self.record_type == RecordType.Scored:
            texts = [record.sentence1 for record in records]
            text_ids = self._ids_tensor(texts)

            texts_pair = [record.sentence2 for record in records]
            text_pair_ids = self._ids_tensor(texts_pair)

            labels = [record.label for record in records]
            labels = torch.tensor(labels, dtype=torch.float32)

            return {
                'text_ids': text_ids,
                'text_pair_ids': text_pair_ids,
                'labels': labels,
            }


class TorchDataset(Dataset):
    def __init__(self, dataset, record_type):
        self.dataset = dataset 
        self.record_cls = record_cls_map[record_type]

    def __getitem__(self, index):
        record = self.dataset[index]
        return self.record_cls(**record) 
    
    def __len__(self):
        return len(self.dataset) 
    
def create_dataloader(dataset, 
                      record_type,
                      tokenizer, 
                      batch_size,
                      max_length=512,
                      num_workers=0,
                      drop_last=False,
                      shuffle=False):
    data_collator = DataCollator(tokenizer, max_length, record_type)
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

    record_type = get_type(train_dataset[0])
    train_ds = TorchDataset(train_dataset, record_type) 
    val_ds = TorchDataset(val_dataset, record_type) if val_dataset else None 

    train_dl = create_dataloader(
        train_ds, 
        record_type,
        tokenizer,
        batch_size,
        shuffle=True, 
        drop_last=True
    )
    val_dl = create_dataloader(
        val_ds, 
        record_type,
        tokenizer,
        batch_size,
        shuffle=False, 
        drop_last=False
    ) if val_ds else None 

    return train_dl, val_dl, record_type 
