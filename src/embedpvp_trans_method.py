import sys
import mowl
mowl.init_jvm("4g")
import torch as th
#import logging
import numpy as np
import pickle as pkl
from mowl.visualization.base import TSNE
import matplotlib.pyplot as plt
from mowl.projection.edge import Edge
from mowl.datasets.builtin import GDADataset, GDAHumanDataset, GDAMouseDataset
from pykeen.models import TransE,ConvE,DistMult,TransR,TransD
from mowl.projection.dl2vec.model import DL2VecProjector 
from mowl.kge import KGEModel

#from mowl.evaluation.rank_based  import EmbeddingsRankBasedEvaluator
from rank_based import EmbeddingsRankBasedEvaluator

from mowl.evaluation.base import TranslationalScore, CosineSimilarity
from mowl.projection.factory import projector_factory, PARSING_METHODS
from util import exist_files, load_pickles, save_pickles
import tempfile
import click as ck
from mowl.datasets.base import PathDataset
import sys
import os
import logging
from get_dis  import GDADataset 
import os.path
from os import path

#--datasets {data} --onto {onto} --vector_size {param['vector_size']} --epochs {param['epochs']} 
# --learning_rate {param['learning_rate']} --batch_size {param['batch_size']}"
ROOT='./mowl_results/'
tmp = tempfile.mkdtemp(prefix='mowl', suffix='/')

@ck.command()
@ck.option("--dataset", "-d", default = "pavs", help = "benchmark dataset: phenobackets,pavs")
@ck.option("--bench", "-d", default = "omim", help = "benchmark type: clinck, omim")
@ck.option("--onto", "-o", default = "hp", help = "Type of test: (go, mp, hp, uberon, union)")
@ck.option("--method", "-m", default = 'TransE')
@ck.option("--vector_size", "-v", type=int, default = 100)
@ck.option("--epochs", "-e", type=int, default = 20)
@ck.option("--learning_rate", "-l", type=float, default = 0.01)
@ck.option("--batch_size", "-bt", default = 100, type=int)
@ck.option("--device", "-d", default = "cuda")

def main(dataset, bench, onto, method, vector_size ,epochs,  learning_rate , batch_size, device):

    owl = f'/encrypted/e3008/Azza/tools/compare/mowl/mowl/data_{onto}/train_{dataset}_{onto}_{bench}.owl'
    test = f'/encrypted/e3008/Azza/tools/compare/mowl/mowl/data_{onto}/test_{onto}.owl'
    valid = f'/encrypted/e3008/Azza/tools/compare/mowl/mowl/data_{onto}/valid_{onto}.owl'
    
    ds = GDADataset(owl,valid, test)
    
    create_dir(ROOT+"graphs/")
    create_dir(ROOT+"walks/")
    create_dir(ROOT+"eval_data/")
    create_dir(ROOT+"results/")
    create_dir(ROOT+"tsne/")
    create_dir(ROOT+"embeddings/")
    create_dir(ROOT+"models/")

    graph_train_file = ROOT + f"graphs/{dataset}_{onto}_{bench}_traint2.pkl"
    graph_test_file = ROOT + f"graphs/{dataset}_{onto}_{bench}_testt2.pkl"
    
    graph_data_files = [graph_train_file, graph_test_file]
    
    walks_file = ROOT +  f"walks/{dataset}_{onto}_{bench}_{method}.txt"
    model_path = ROOT +  f"models/{dataset}_{onto}_{bench}_{method}.pt"

    eval_train_file = ROOT + f"eval_data/{dataset}_{onto}_{bench}_training_sett2.pkl"
    eval_test_file = ROOT + f"eval_data/{dataset}_{onto}_{bench}_test_sett2.pkl"
    eval_heads_file = ROOT + f"eval_data/{dataset}_{onto}_{bench}_head_entitiest2.pkl"
    eval_tails_file = ROOT + f"eval_data/{dataset}_{onto}_{bench}_tail_entitiest2.pkl"


    if exist_files(*graph_data_files):
        logging.info("Graph found. Loading...")
        train_edges, test_edges, *_ = load_pickles(*graph_data_files)

        #print(test_edges)
        #print(Edge.get_entities_and_relations(test_edges))
        # 'http://default/mowl/OMIM:616409'}, {'http://www.w3.org/2002/07/is_associated_with'}
        #for i in test_edges:
        #    print(type(i))
        #    print(i.as_str)
            
        #exit()
    else:
        logging.info("Graph not found. Generating...")    
        proj = DL2VecProjector(True)
        train_edges = proj.project(ds.ontology)
        test_edges = proj.project(ds.testing)
        save_pickles((train_edges, graph_train_file), (test_edges, graph_test_file))

    eval_data_files = [eval_train_file, eval_test_file, eval_heads_file, eval_tails_file]
    
    if exist_files(*eval_data_files):
        logging.info("Evaluation data found. Loading...")
        eval_train_edges, eval_test_edges, head_entities, tail_entities = load_pickles(*eval_data_files)
        
    else:
        logging.info("Evaluation data not found. Generating...")

        proj = DL2VecProjector(True)
        eval_train_edges = proj.project(ds.ontology)
        eval_test_edges = proj.project(ds.testing) 
              
        test_entities, _ = Edge.getEntitiesAndRelations(eval_test_edges)
        head_entities, tail_entities = ds.evaluation_class() 

        save_pickles(
                (eval_train_edges, eval_train_file),
                (eval_test_edges, eval_test_file),
                (head_entities, eval_heads_file),
                (tail_entities, eval_tails_file)
            )

    classes, relations = Edge.getEntitiesAndRelations(train_edges)
    classes, relations = list(set(classes)), list(set(relations))
    classes.sort()
    relations.sort()
    class_to_id = {c: i for i, c in enumerate(classes)}
    rel_to_id = {r: i for i, r in enumerate(relations)}

    triples_factory = Edge.as_pykeen(train_edges, create_inverse_triples = True, entity_to_id = class_to_id, relation_to_id=rel_to_id)

    #ConvE DistMult TransE TransR TransD
    if method == 'TransD':
        pk_model = TransD(triples_factory=triples_factory, embedding_dim = vector_size, random_seed=42)

    elif method == 'ConvE':
        pk_model = ConvE(triples_factory=triples_factory, embedding_dim = vector_size, random_seed=42)

    elif method == 'DistMult':
        pk_model = DistMult(triples_factory=triples_factory, embedding_dim = vector_size, random_seed=42)

    elif method == 'TransR':
        pk_model = TransR(triples_factory=triples_factory, embedding_dim = vector_size, random_seed=42)

    elif method == 'TransE':
        pk_model = TransE(triples_factory=triples_factory, embedding_dim = vector_size, random_seed=42)

    model = KGEModel(triples_factory, pk_model, epochs = epochs, batch_size = batch_size, lr=learning_rate, device = device, model_filepath=model_path) #30, 1000 was good also 20 - 500


    model.train()
        
    model.load_best_model()
    ent_embs = model.class_embeddings_dict
    rel_embs = model.object_property_embeddings_dict

    with open(ROOT+f'embeddings/{dataset}_{onto}_{bench}_{method}_embed.pkl', "wb") as f:
        pkl.dump(ent_embs, f)
        
    with open(ROOT+f'embeddings/{dataset}_{onto}_{bench}_{method}_rel.pkl', "wb") as f:
        pkl.dump(rel_embs, f)

    #my_genes_head=set()
    #file_name = f'/encrypted/e3008/Azza/tools/compare/mowl/mowl/data/{dataset}_annot.tsv'
    #my_dis_tail=set()
    #with open(file_name) as f:
    #    for row in f:
    #        my_dis_tail.add('http://ontology.com/'+row.split()[0])
    
    #with open('/encrypted/e3008/Azza/tools/EmbedPVP/NEW/scripts/benchmark/embed/ben_genes.pkl', 'rb') as f:
    #    my_genes_head = pkl.load(f)

    score_func = lambda x: - model.score_method_tensor(x)
    evaluator = EmbeddingsRankBasedEvaluator(
                ent_embs,
                test_edges, #test_edges
                TranslationalScore, #TranslationalScore,
                score_func =  score_func, #model.score_method_tensor, #score_func
                training_set= train_edges,
                relation_embeddings = rel_embs,
                head_entities = head_entities, #genens for pavs head_entities
                tail_entities = tail_entities, #dis pavs id tail_entities
                device = device
    )

    evaluator.evaluate(activation = th.sigmoid, show=False)
    tex_table = ""
    for k, v in evaluator.metrics.items():
        tex_table += f"{v} &\t"
        print(f"{k}\t{v}\n")
    print(f"\n{tex_table}")

    file_prediction =ROOT+f"results/{dataset}_{onto}_{bench}_{method}_gd.pkl"
    final={}
    u = f"{dataset}_{onto}_{bench}_{method}"
    final[u] = evaluator.metrics
    print(final)

    with open(file_prediction,'wb') as f:
        pkl.dump(final, f)


def create_dir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


if __name__ == "__main__":
    main()
    
