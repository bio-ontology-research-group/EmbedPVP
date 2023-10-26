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
from ELEmModule import ELEmModule, ELEmbeddings, ELBoxEmbeddings
from mowl.nn.elmodule import ELModule
#from mowl.evaluation.rank_based import ModelRankBasedEvaluator
from rank_based import ModelRankBasedEvaluator


from mowl.projection.dl2vec.model import DL2VecProjector 
from mowl.kge import KGEModel
from mowl.evaluation.rank_based import EmbeddingsRankBasedEvaluator
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


ROOT='./mowl_results/'
tmp = tempfile.mkdtemp(prefix='mowl', suffix='/')

@ck.command()
@ck.option("--dataset", "-d", default = "pavs", help = "benchmark dataset: phenobackets,pavs")
@ck.option("--bench", "-d", default = "omim", help = "benchmark type: clinck, omim")
@ck.option("--onto", "-o", default = "hp", help = "Type of test: (go, mp, hp, uberon, union)")
@ck.option("--method", "-m", default = 'TransE')
@ck.option("--vector_size", "-v", type=int, default = 100)
@ck.option("--margin", "-m", type=float, default = 0.1)
@ck.option("--neg_norm", "-n", default = 1.0)
@ck.option("--epochs", "-e", type=int, default = 20)
@ck.option("--learning_rate", "-l", type=float, default = 0.01)
@ck.option("--batch_size", "-bt", default = 100, type=int)
@ck.option("--device", "-d", default = "cuda")

def main(dataset, bench, onto, method, vector_size , margin , neg_norm, epochs,  learning_rate , batch_size, device):

    owl = f'/encrypted/e3008/Azza/tools/compare/mowl/mowl/data_{onto}/train_{dataset}_{onto}_{bench}.owl'
    test = f'/encrypted/e3008/Azza/tools/compare/mowl/mowl/data_{onto}/test_{onto}.owl'
    valid = f'/encrypted/e3008/Azza/tools/compare/mowl/mowl/data_{onto}/valid_{onto}.owl'
      
    ds = GDADataset(owl, valid, test)

    create_dir(ROOT+"graphs/")
    create_dir(ROOT+"walks/")
    create_dir(ROOT+"eval_data/")
    create_dir(ROOT+"results/")
    create_dir(ROOT+"tsne/")
    create_dir(ROOT+"embeddings/")
    create_dir(ROOT+"models/")

    graph_train_file = ROOT + f"graphs/{dataset}_{onto}_{bench}_train.pkl"
    graph_test_file = ROOT + f"graphs/{dataset}_{onto}_{bench}_test.pkl"
    graph_data_files = [graph_train_file, graph_test_file]
    
    walks_file = ROOT +  f"walks/{dataset}_{onto}_{bench}_{method}.txt"
    model_path = ROOT +  f"models/{dataset}_{onto}_{bench}_{method}.pt"

    eval_train_file = ROOT + f"eval_data/{dataset}_{onto}_{bench}_training_set.pkl"
    eval_test_file = ROOT + f"eval_data/{dataset}_{onto}_{bench}_test_set.pkl"
    eval_heads_file = ROOT + f"eval_data/{dataset}_{onto}_{bench}_head_entities.pkl"
    eval_tails_file = ROOT + f"eval_data/{dataset}_{onto}_{bench}_tail_entities.pkl"


    if exist_files(*graph_data_files):
        logging.info("Graph found. Loading...")
        train_edges, test_edges, *_ = load_pickles(*graph_data_files)

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

    if method == 'el':
        model = ELEmbeddings(ds,
                             embed_dim=vector_size,
                             margin=margin,
                             reg_norm=neg_norm,
                             learning_rate=learning_rate,
                             epochs=epochs,
                             batch_size=batch_size,
                             model_filepath=model_path,
                             device=device)
                             

    elif method == 'elbox':
        model = ELBoxEmbeddings(ds,
                             embed_dim=vector_size,
                             margin=margin,
                             reg_norm=neg_norm,
                             learning_rate=learning_rate,
                             epochs=epochs,
                             batch_size=batch_size,
                             model_filepath=model_path,
                             device=device)
                                      
    model.train()
    model.load_best_model()
    
    ent_embs, rel_embs = model.get_embeddings()

    with open(ROOT+f'embeddings/{dataset}_{onto}_{bench}_{method}_embed.pkl', "wb") as f:
        pkl.dump(ent_embs, f)
        
    with open(ROOT+f'embeddings/{dataset}_{onto}_{bench}_{method}_rel.pkl', "wb") as f:
        pkl.dump(rel_embs, f)


    with th.no_grad():
        model.load_best_model()
        evaluator = ModelRankBasedEvaluator(
                    model,
                    device=device,
                    eval_method=model.model.gci2_loss
        )
        evaluator.evaluate(show=True)

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
    
