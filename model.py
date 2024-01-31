import torch 
from data import RecordType 

class CoSentLoss(torch.nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.temp = temp 

    def forward(self, embeddings1, embeddings2, embeddings3=None, labels=None):
        if labels is not None:
            labels = labels.to(embeddings1.device)

            predict_similarity = torch.cosine_similarity(
                embeddings1,
                embeddings2,
                dim=-1,
            )
            predict_similarity = predict_similarity / self.temp 

            cosine_similarity_diff = -(predict_similarity.unsqueeze(0) - predict_similarity.unsqueeze(1))
            smaller_mask = labels.unsqueeze(0) <= labels.unsqueeze(1)
            cosine_similarity_diff = cosine_similarity_diff.masked_fill(smaller_mask, -1e12)

            log1 = torch.tensor([0.0]).to(embeddings1.device)
            logits = torch.cat((cosine_similarity_diff.view(-1), log1))
            loss = torch.logsumexp(logits, dim=0)
        elif embeddings3 is not None:
            sim_pos_vector = torch.cosine_similarity(
                embeddings1, 
                embeddings2, 
                dim=-1, 
            )
            sim_neg_matrix = torch.cosine_similarity(
                embeddings1.unsqueeze(1),
                embeddings3.unsqueeze(0),
                dim=-1, 
            )
            sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
            sim_matrix = sim_matrix / self.temp 
            sim_matrix_diff = sim_matrix - sim_matrix[:, 0].unsqueeze(1)
            loss = torch.logsumexp(sim_matrix_diff, dim=1).mean() 
        else:
            sim_matrix = torch.cosine_similarity(
                embeddings1.unsqueeze(1),
                embeddings2.unsqueeze(0),
                dim=-1,
            )
            sim_matrix = sim_matrix / self.temp 
            sim_matrix_diag = sim_matrix.diag()
            sim_matrix_diff = sim_matrix - sim_matrix_diag.unsqueeze(1)
            loss = torch.logsumexp(sim_matrix_diff, dim=1).mean()

        return loss 
    

class TorchModel(torch.nn.Module):
    def __init__(self, model, record_type, temp=0.05):
        super().__init__()
        self.model = model 
        self.record_type = record_type 
        self.criterion = CoSentLoss(temp)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)

    def _get_embeddings(self, tensor):
        input = {
            'input_ids': tensor.to(self.device),
            'attention_mask': (tensor != self.model.tokenizer.pad_token_id).to(self.device)
        }
        output = self.model.forward(input)
        return output['sentence_embedding']

    def forward(self, batch):
        if self.record_type == RecordType.Pair:
            embeddings1 = self._get_embeddings(batch['text_ids'])
            embeddings2 = self._get_embeddings(batch['text_pos_ids'])
            loss = self.criterion(embeddings1, embeddings2)
        elif self.record_type == RecordType.Triplet:
            embeddings1 = self._get_embeddings(batch['text_ids'])
            embeddings2 = self._get_embeddings(batch['text_pos_ids'])
            embeddings3 = self._get_embeddings(batch['text_neg_ids'])
            loss = self.criterion(embeddings1, embeddings2, embeddings3=embeddings3)
        elif self.record_type == RecordType.Scored:
            embeddings1 = self._get_embeddings(batch['text_ids'])
            embeddings2 = self._get_embeddings(batch['text_pair_ids'])
            loss = self.criterion(embeddings1, embeddings2, labels=batch['labels'])
        else:
            loss = None 

        return {'loss': loss} 
