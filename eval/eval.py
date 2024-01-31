import math
import numpy as np
from tqdm import tqdm
from sentence_transformers import util
from datasets import load_from_disk
#https://huggingface.co/datasets/C-MTEB/T2Retrieval
#https://huggingface.co/datasets/C-MTEB/T2Retrieval-qrels

def calc_IDCG(correlations):
    corrs = sorted(correlations, reverse=True)
    acc = 0
    for i in range(len(corrs)):
        if corrs[i]: acc += 1 / math.log2(i+2)
        else: break
    return acc

def calc_DCG(correlations):
    acc = 0
    for i in range(len(correlations)):
        if correlations[i]: acc += 1 / math.log2(i+2)
    return acc

def calc_NDCG(correlations):
    dcg = calc_DCG(correlations)
    idcg = calc_IDCG(correlations)
    return dcg / max(idcg, 1e-9)

def eval_st_model(st_model, logger, topk=10):
    logger.info(f'eval loading data ......')
    dataset = load_from_disk('eval/t2retrieval')
    logger.info(f'eval data loaded ......')
    
    relevant_docs = {}
    for qrels_data in dataset['dev']:
        id1 = qrels_data['qid']
        id2 = qrels_data['pid']
        d = relevant_docs.get(str(id1), {})
        d[str(id2)] = 1
        relevant_docs[str(id1)] = d 

    q_ids, q_m = [], []
    for queries_data in tqdm(dataset['queries']):
        id = queries_data['id']
        text = queries_data['text']
        embed = st_model.encode(text)
        q_ids.append(id)
        q_m.append(embed)
    q_m = np.array(q_m)

    c_ids, c_m = [], []
    for corpus_data in tqdm(dataset['corpus']):
        id = corpus_data['id']
        text = corpus_data['text']
        embed = st_model.encode(text)
        c_ids.append(id)
        c_m.append(embed)
    c_m = np.array(c_m)

    logger.info(f'q_m shape: {q_m.shape}, c_m shape: {c_m.shape}')
    sim = util.cos_sim(q_m, c_m)
    logger.info(f'sim shape: {sim.shape}')
    index = np.argsort(sim, axis=-1).numpy()[:,::-1]

    arr = []
    for i in tqdm(range(len(index))):
        corr = []
        q_id = q_ids[i]
        for j in range(topk):
            c_id = c_ids[index[i][j]]
            if c_id in relevant_docs[q_id].keys():
                corr.append(1)
            else:
                corr.append(0)
        arr.append(calc_NDCG(corr))

    return sum(arr) / len(arr)
