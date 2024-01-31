import os 
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings 
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from datasets import load_from_disk
from mteb import MTEB 
from mteb.abstasks import AbsTaskRetrieval

#https://huggingface.co/datasets/C-MTEB/T2Retrieval
#https://huggingface.co/datasets/C-MTEB/T2Retrieval-qrels
class T2Retrieval(AbsTaskRetrieval): 
    @property 
    def description(self):
        return {
            'name': 'T2Retrieval',
            'description': 'retrieval dataset',
            'category': 's2p',
            'type': 'Retrieval',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }
    
    def load_data(self, **kwargs):
        print('kwargs:', kwargs)
        self.dataset = load_from_disk('eval/t2retrieval')
        self.data_loaded = True 

        self.corpus, self.queries, self.relevant_docs = {'test': {}}, {'test': {}}, {'test': {}}
        for corpus_data in self.dataset['corpus']:
            id = corpus_data['id']
            text = corpus_data['text']
            self.corpus['test'][str(id)] = {'title': '', 'text': text}

        for queries_data in self.dataset['queries']:
            id = queries_data['id']
            text = queries_data['text']
            self.queries['test'][str(id)] = text

        for qrels_data in self.dataset['dev']:
            id1 = qrels_data['qid']
            id2 = qrels_data['pid']
            d = self.relevant_docs['test'].get(str(id1), {})
            d[str(id2)] = 1
            self.relevant_docs['test'][str(id1)] = d


def eval_st_model(st_model, output_folder='eval_output', batch_size=8):
    evaluation = MTEB(tasks=[T2Retrieval()]) 
    eval_res = evaluation.run(st_model, output_folder=output_folder, batch_size=batch_size)
    return eval_res['T2Retrieval']['test']
