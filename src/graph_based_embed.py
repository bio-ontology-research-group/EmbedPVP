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
from rank_basedgraph import EmbeddingsRankBasedEvaluator

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

from mowl.projection import DL2VecProjector, OWL2VecStarProjector
from mowl.walking import DeepWalk
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
#from mowl.evaluation.rank_based import EmbeddingsRankBasedEvaluator

from mowl.evaluation.base import CosineSimilarity
from mowl.projection import TaxonomyWithRelationsProjector


ROOT='./mowl_results/'
tmp = tempfile.mkdtemp(prefix='mowl', suffix='/')
print(tmp)


#final_graph_based.py --onto hp --method owl2vec --vector_size 100 --window 5  --epochs 50 --num_walks 50 --num_walks 50 --walk_length 30

@ck.command()
#@ck.option("--datasets", "-d", default = "pavs", help = "benchmark dataset: phenobackets,pavs")
@ck.option("--onto", "-o", default = "hp", help = "Type of test: (go, mp, hp, uberon, union)")
@ck.option("--method", "-m", default = 'TransE')
@ck.option("--vector_size", "-v", type=int, default = 100)
@ck.option("--window", "-v", type=int, default = 5)
@ck.option("--epochs", "-e", type=int, default = 20)
@ck.option("--num_walks", "-e", type=int, default = 10)
@ck.option("--walk_length", "-e", type=int, default = 30)
@ck.option("--device", "-d", default = "cpu")

def main(onto, method, vector_size ,window, epochs,  num_walks , walk_length, device):

    owl = f'/encrypted/e3008/Azza/tools/compare/mowl/mowl/data_{onto}/train_{onto}.owl'
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

    graph_train_file = ROOT + f"graphs/{onto}_{method}_train.pkl"
    graph_test_file = ROOT + f"graphs/{onto}_{method}_test.pkl"
    graph_data_files = [graph_train_file, graph_test_file]
    
    walks_file = tmp +  f"{onto}_{method}.txt"

    eval_train_file = ROOT + f"eval_data/{onto}_{method}_training_set.pkl"
    eval_test_file = ROOT + f"eval_data/{onto}_{method}_test_set.pkl"
    eval_heads_file = ROOT + f"eval_data/{onto}_{method}_head_entities.pkl"
    eval_tails_file = ROOT + f"eval_data/{onto}_{method}_tail_entities.pkl"

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

        #ds_eval = GDAMouseDataset()
        #ds_eval_train_edges = proj.project(ds_eval.ontology)
        #ds_eval_test_edges = proj.project(ds_eval.testing)

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
    if method == 'dl2vec':
        projector = DL2VecProjector(True)
        train_edges = projector.project(ds.ontology)
        test_edges = projector.project(ds.testing)
        
        
    elif method == 'owl2vc':
        projector = OWL2VecStarProjector(True)
        train_edges = projector.project(ds.ontology)
        test_edges = projector.project(ds.testing)
      

    walker = DeepWalk(num_walks, # number of walks per node
                      walk_length, # walk length
                      0.1, # restart probability
                      workers=10, outfile = tmp+'walk',seed=42) # number of threads

    walks = walker.walk(train_edges)
    walks_file = walker.outfile
    sentences = LineSentence(walks_file)
    model = Word2Vec(sentences, vector_size=vector_size, epochs = epochs, window=window, min_count=1, workers=10)


    genes, diseases = ds.evaluation_class()
    projector = TaxonomyWithRelationsProjector(taxonomy=False,
                                               relations=["http://is_associated_with"])

    assert len(test_edges) > 0

    vectors = model.wv
    evaluator = EmbeddingsRankBasedEvaluator(
        vectors,
        test_edges,
        CosineSimilarity,
        training_set=train_edges,
        head_entities = genes,
        tail_entities = diseases,
        device = device
    )
  
    evaluator.evaluate(show=False)
    tex_table = ""
    for k, v in evaluator.metrics.items():
        tex_table += f"{v} &\t"
        print(f"{k}\t{v}\n")
    print(f"\n{tex_table}")

    allparam=f"{onto}_{method}_{vector_size}_{epochs}_{window}_{num_walks}_{walk_length}"
    final={}
    final[allparam] = evaluator.metrics
    print(final)

    filename=ROOT+'params/'+str(allparam)+'.pkl'
    with open(filename,'wb') as f:
        pkl.dump(final, f)

def create_dir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)


if __name__ == "__main__":
    main()
    
