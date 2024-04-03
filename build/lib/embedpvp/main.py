#!/usr/bin/env python
#!/usr/bin/env python
# Standard Library Imports
import os
import sys
import json
import gzip
import logging
import subprocess
import argparse
import multiprocessing as mp
from threading import Lock
from random import seed
from statistics import mean
from operator import itemgetter
import warnings
import pdb
import tempfile
import shutil
import math

import click as ck
import numpy as np
import pandas as pd
import gensim
import random
import pickle as pkl
from pandas.api.types import CategoricalDtype
import sklearn.metrics as metrics
from progress.bar import Bar

import mowl
import time
from scipy.stats import rankdata
from argparse import ArgumentParser
import networkx as nx
# Initialize JVM
mowl.init_jvm("20g")
from mowl.datasets.base import PathDataset
from mowl.walking import DeepWalk
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import scipy.stats as ss
from mowl.ontology.extend import insert_annotations
from mowl.ontology.create import create_from_triples
import torch as th
from mowl.projection.edge import Edge
from mowl.datasets.builtin import GDADataset, GDAHumanDataset, GDAMouseDataset
from pykeen.models import TransE, ConvE, DistMult, TransR, TransD
from mowl.projection.dl2vec.model import DL2VecProjector
from mowl.kge import KGEModel
from mowl.evaluation.base import TranslationalScore, CosineSimilarity
from embedpvp.ELEmModule import ELEmModule, ELEmbeddings, ELBoxEmbeddings
from mowl.projection.factory import projector_factory, PARSING_METHODS
from mowl.projection import DL2VecProjector, OWL2VecStarProjector
from mowl.projection import TaxonomyWithRelationsProjector
from mowl.nn import ELModule
from pykeen.triples import TriplesFactory
from gensim.models import KeyedVectors
from embedpvp.get_dis  import GDADataset 


# Set random seed
np.random.seed(42)


# Set the logging level for the OWL API package to WARNING
logging.getLogger("uk.ac.manchester.cs.owl.owlapi").setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# Pandas options
pd.options.mode.chained_assignment = None  # default='warn'

# Threading lock
lock = Lock()

# Logger configuration
logger = logging.getLogger('my-logger')
logger.propagate = False
logger.disabled = False

# Temporary directory
tmp = tempfile.mkdtemp(prefix='EmbedPVP', suffix='/')

@ck.command()
@ck.option('--data-root', '-d', default='data/', help='Data root folder', required=True)
@ck.option('--in_file', '-i', help='Annotated Input file', required=True)
@ck.option('--pathogenicity', '-p', help='Path to the pathogenicity prediction file (CADD)', required=True)
@ck.option('--hpo','-hpo', help='List of phenotype codes separated by commas', required=True)
@ck.option('--model_type','-m', help='Ontology model, one of the following (go , mp , hp, uberon, union)', default='hp')
@ck.option('--embedding','-e', help='Preferred embedding model (e.g. TransD, TransE, TranR, ConvE ,DistMult, DL2vec, OWL2vc)', default='dl2vec', required=True)
@ck.option('--outdir','-dir', default='output/', help='Path to the output directory')
@ck.option('--outfile','-o', default='embedpvp_results.tsv', help='Path to the results output file')

def main(data_root, in_file, pathogenicity, hpo, model_type, embedding, outdir, outfile):
    # Check data folder and required files
    try:
        if os.path.exists(data_root):
            in_file = os.path.join(data_root, in_file)
            owl_file = os.path.join(data_root, f'mowl_{model_type}.owl')
            pathogenicity_scores = os.path.join(data_root, pathogenicity)
            if not os.path.exists(in_file):
                raise Exception(f'Annotated input file ({in_file}) is missing!')
            if not os.path.exists(owl_file):
                raise Exception(f'Model file ({owl_file}) is missing!')
            if not os.path.exists(pathogenicity_scores):
                raise Exception(f'Pathogenicity prediction file ({pathogenicity_scores}) is missing!')
        else:
            raise Exception(f'Data folder ({data_root}) does not exist!')
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    # Initialize progress bar
    bar = Bar(max=4, fill='=', suffix='%(percent)d%%')

    state = 'START'
    while state != 'DONE':
        bar.next()
        ontology = mowl_insert_annotations(hpo, owl_file, outdir)
        print(f" Insert annotations to the ({model_type}) ontology...")

        if not os.path.exists(f'{outdir}/ontology.owl'):
            raise Exception(f'Ontology file with the inserted annotation not found ({outdir}/ontology.owl) is missing!')

        bar.next()
        print(" Phenotype prediction...")
        variants_features = genomic_features(data_root, in_file, pathogenicity_scores)
        
        embedding_type = embedding.lower()
        
        if embedding_type in ['dl2vec', 'owl2vec']:
            ds = PathDataset(f'{outdir}/ontology.owl', validation_path=None, testing_path=None)
            embeddings = graph_based_embeddings(ds, embedding_type, outdir)
            bar.next()
            results = calculate_graph_embed_similarity(variants_features, embeddings)

        elif embedding_type in ['el', 'elbox']:
            ds = GDADataset(f'{outdir}/ontology.owl', f'{data_root}/valid_{model_type}.owl', f'{data_root}/test_{model_type}.owl') #validation_path=None, testing_path=None)
            model, class_to_id, rel_to_id= semantic_embeddings(ds, embedding_type, outdir)
            bar.next()
            results = calculate_semantic_model_similarity(model, variants_features, class_to_id, rel_to_id)

        elif embedding_type in ['transd', 'transe', 'tranr', 'conve', 'distmult']:
            ds = PathDataset(f'{outdir}/ontology.owl', validation_path=None, testing_path=None)
            model, class_to_id, rel_to_id= translation_embeddings(ds, embedding_type, outdir)
            bar.next()
            results = calculate_translation_embed_similarity(model, variants_features, class_to_id, rel_to_id)

        print(" Variants prediction...")
        
        out_file = os.path.join(outdir, outfile)
        results.to_csv(out_file, sep='\t', index=False)
        if not os.path.exists(out_file):
            raise Exception(f'The output file does not exist!')
        else:
            # Write the results to a file
            bar.next()
            print(f' DONE! You can find the prediction results in the output file: {outdir}/{outfile}')

        state = 'DONE'

# decorater used to block function printing to the console
def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper

@blockPrinting
def mowl_insert_annotations(hpo, owl_file, output_directory):
    """
    Insert annotations into an ontology.

    Args:
        data_root (str): Root directory for data.
        hpo (str): Comma-separated list of HPO terms.
        owl_file (str): Path to the OWL file.
        output_directory (str): Directory for output files.

    This function inserts annotations into the ontology specified by `owl_file`.
    """
    with open(f'{output_directory}/pheno.txt', "w") as fp:
        fp.write('patient' + "\t")
        for pheno in hpo.split(','):
            fp.write("http://purl.obolibrary.org/obo/" + pheno.replace(':', '_') + "\t")

    print("Reading the input phenotypes...")

    # Define annotations for ontology construction
    annotations = [(f'{output_directory}/pheno.txt', "http://has_annotation", True)]

    # Insert annotations into the ontology
    insert_annotations(owl_file, annotations, f'{output_directory}/ontology.owl')

@blockPrinting
def graph_based_embeddings(ds, method, outputdir, vector_size=100, window=5, epochs=1, num_walks=1, walk_length=30):
    """
    Generate graph-based embeddings using DeepWalk or similar methods.

    Args:
        ds: Your dataset or ontology object.
        method (str): The method to use ('dl2vec' or 'owl2vec').
        outputdir (str): The directory for saving output files.
        vector_size (int): The size of the word vectors.
        window (int): The size of the context window.
        epochs (int): The number of training epochs.
        num_walks (int): The number of walks per node.
        walk_length (int): The length of each walk.

    Returns:
        gensim.models.KeyedVectors: Word vectors.
    """
    walks_file = f'{outputdir}/{method}_walks.txt'

    if method == 'dl2vec':
        projector = DL2VecProjector(True)
        train_edges = projector.project(ds.ontology)
    elif method == 'owl2vec':
        projector = OWL2VecStarProjector(True)
        train_edges = projector.project(ds.ontology)

    walker = DeepWalk(num_walks=num_walks, walk_length=walk_length, alpha=0.1, workers=10, outfile=walks_file, seed=42)
    walks = walker.walk(train_edges)
    sentences = LineSentence(walks_file)

    model = Word2Vec(sentences, vector_size=vector_size, epochs=epochs, window=window, min_count=1, workers=10)
    word2vec_file = f'{outputdir}/{method}_embed.pkl'
    model.save(word2vec_file)

    vectors = model.wv
    
    return vectors

@blockPrinting
def semantic_embeddings(ds, method, outputdir, vector_size=100, margin=0.1, neg_norm=1.0, epochs=10, learning_rate=0.01, batch_size=4096 * 8, device="cpu"):
    """
    Generate semantic embeddings using ELEmbeddings or ELBoxEmbeddings methods.

    Args:
        ds: Your dataset or ontology object.
        method (str): The method to use ('el' or 'elbox').
        outputdir (str): The directory for saving output files.
        vector_size (int): The size of the embedding vectors.
        margin (float): Margin value for loss function.
        neg_norm (float): Regularization norm for negative samples.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for training.
        batch_size (int): Batch size for training.
        device (str): Device for training (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple of numpy arrays: Entity and relation embeddings.
    """
    model_path = f"{outputdir}/{method}.pt"
    proj = DL2VecProjector(True)
    train_edges = proj.project(ds.ontology)
    
    # Extract unique classes and relations
    classes, relations = Edge.getEntitiesAndRelations(train_edges)
    classes, relations = list(set(classes)), list(set(relations))
    classes.sort()
    relations.sort()
    class_to_id = {c: i for i, c in enumerate(classes)}
    rel_to_id = {r: i for i, r in enumerate(relations)}

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
    #ent_embs, rel_embs = model.get_embeddings()
    #with open(f'{outputdir}/{method}_embed.pkl', "wb") as f:
    #    pkl.dump(ent_embs, f)
    #with open(f'{outputdir}/{method}_rel.pkl', "wb") as f:
    #    pkl.dump(rel_embs, f)
        
    return model, class_to_id, rel_to_id

@blockPrinting
def translation_embeddings(ds, method, outputdir, vector_size=50, epochs=1, learning_rate=0.01, batch_size=10000, device="cpu"):
    """
    Generate translation-based embeddings using various models.

    Args:
        ds: Your dataset or ontology object.
        method (str): The method to use ('TransD', 'ConvE', 'DistMult', 'TransR', 'TransE').
        outputdir (str): The directory for saving output files.
        vector_size (int): The size of the embedding vectors.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for training.
        batch_size (int): Batch size for training.
        device (str): Device for training (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple of dictionaries: Entity and relation embeddings.
    """
    proj = DL2VecProjector(True)
    train_edges = proj.project(ds.ontology)
    model_path = f"{outputdir}/{method}.pt"

    # Extract unique classes and relations
    classes, relations = Edge.getEntitiesAndRelations(train_edges)
    classes, relations = list(set(classes)), list(set(relations))
    classes.sort()
    relations.sort()
    class_to_id = {c: i for i, c in enumerate(classes)}
    rel_to_id = {r: i for i, r in enumerate(relations)}

    # Create a PyKEEN TriplesFactory
    triples_factory = Edge.as_pykeen(train_edges, create_inverse_triples = True, entity_to_id = class_to_id, relation_to_id=rel_to_id)

    # Initialize the PyKEEN model based on the selected method
    if method == 'transd':
        pk_model = TransD(
            triples_factory=triples_factory,
            embedding_dim=vector_size,
            random_seed=42,
        )
    elif method == 'conve':
        pk_model = ConvE(
            triples_factory=triples_factory,
            embedding_dim=vector_size,
            random_seed=42,
        )
    elif method == 'distmult':
        pk_model = DistMult(
            triples_factory=triples_factory,
            embedding_dim=vector_size,
            random_seed=42,
        )
    elif method == 'transr':
        pk_model = TransR(
            triples_factory=triples_factory,
            embedding_dim=vector_size,
            random_seed=42,
        )
    elif method == 'transe':
        pk_model = TransE(
            triples_factory=triples_factory,
            embedding_dim=vector_size,
            random_seed=42,
        )

    # Create a KGEModel and train the model
    model = KGEModel(triples_factory, pk_model, epochs = epochs, batch_size = batch_size, lr=learning_rate, device = device, model_filepath=model_path) #30, 1000 was good also 20 - 500

    model.train()
    model.load_best_model()
    #ent_embs = model.class_embeddings_dict
    #rel_embs = model.object_property_embeddings_dict

    # Save embeddings to files
    #with open(f'{outputdir}/{method}_embed.pkl', "wb") as f:
    #    pkl.dump(ent_embs, f)

    #with open(f'{outputdir}/{method}_rel.pkl', "wb") as f:
    #    pkl.dump(rel_embs, f)

    return model, class_to_id, rel_to_id

                
def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

@blockPrinting
def calculate_graph_embed_similarity(features_dict, embeddings, weight=0.6):
    final_scores = {}
    
    for variant_id, variant_data in features_dict.items():
        final_scores[variant_id] = variant_data
        cadd_score = variant_data['CADD']
        gene_scores = []
        #print(variant_data)
        
        for gene in variant_data['allgens']:
            patient = 'http://ontology.com/patient'
            gene_entity = f'http://ontology.com/{gene}'
            if patient in embeddings.index_to_key and gene_entity in embeddings.index_to_key:
                patient_id = embeddings[patient]
                gene_id = embeddings[gene_entity]
                gene_score = cosine_similarity(gene_id,patient_id)
                normalized_similarity = (gene_score + 1) / 2
                gene_scores.append(normalized_similarity)
                
        if len(gene_scores)>0:
            max_gene_score = max(gene_scores)
        else:
            max_gene_score = 0
            
        gene_scores_weighted = (weight * max_gene_score) + ((1 - weight) * cadd_score)
        final_scores[variant_id]['EmbedPVP_Score'] = gene_scores_weighted
                
    result_df = pd.DataFrame.from_dict(final_scores, orient='index')
    del result_df['allgens'] , result_df['varID'] , result_df['CADD']
    result_df = result_df.sort_values(by='EmbedPVP_Score', ascending=False)
    result_df['Rank'] = result_df['EmbedPVP_Score'].rank(method='min', ascending=False)
    result_df = result_df.drop_duplicates()

    return result_df

@blockPrinting
def calculate_semantic_model_similarity(model, features_dict, class_to_id, rel_to_id, weight=0.6):    
    final_scores = {}
    relation = 'http://is_associated_with'
    
    for variant_id, variant_data in features_dict.items():
        final_scores[variant_id] = variant_data
        cadd_score = variant_data['CADD']
        gene_scores = []
                
        for gene in variant_data['all_genes']:
            patient = 'http://ontology.com/patient'
            gene_entity = f'http://ontology.com/{gene}'
            if patient in class_to_id and gene_entity in class_to_id:
                patient_id = class_to_id[patient]
                gene_id = class_to_id[gene_entity]
                relation_id = rel_to_id[relation]
                triple = (gene_id, relation_id, patient_id)
                triple_tensor = th.tensor(triple)
                triple_tensor = triple_tensor.reshape(1, 3)                 
                score = model.model.gci2_loss(triple_tensor)
                score = (score.item()) / 100
                gene_scores.append(score)
        
        if len(gene_scores)>0:
            max_gene_score = max(gene_scores)
        else:
            max_gene_score = 0
        gene_scores_weighted = (weight * max_gene_score) + ((1 - weight) * cadd_score)
        final_scores[variant_id]['EmbedPVP_Score'] = gene_scores_weighted
                
    result_df = pd.DataFrame.from_dict(final_scores, orient='index')
    del result_df['all_genes'], result_df['varID'], result_df['CADD']
    result_df = result_df.sort_values(by='EmbedPVP_Score', ascending=False)
    result_df['Rank'] = result_df['EmbedPVP_Score'].rank(method='min', ascending=False)
    result_df = result_df.drop_duplicates()
    
    return result_df

@blockPrinting                        
def calculate_translation_embed_similarity(model, features_dict, class_to_id, rel_to_id, weight=0.6):
    final_scores = {}
    relation = 'http://is_associated_with'
    
    for variant_id, variant_data in features_dict.items():
        final_scores[variant_id] = variant_data
        cadd_score = variant_data['CADD']
        gene_scores = []
                
        for gene in variant_data['allgens']:
            patient = 'http://ontology.com/patient'
            gene_entity = f'http://ontology.com/{gene}'
            if patient in class_to_id and gene_entity in class_to_id:
                patient_id = class_to_id[patient]
                gene_id = class_to_id[gene_entity]
                relation_id = rel_to_id[relation]
                triple = (gene_id, relation_id, patient_id)
                score = model.score_method_point(triple)
                score = (score.item()) / 100
                gene_scores.append(score)
                
        if len(gene_scores)>0:
            max_gene_score = max(gene_scores)
        else:
            max_gene_score = 0
        gene_scores_weighted = (weight * max_gene_score) + ((1 - weight) * cadd_score)
        final_scores[variant_id]['EmbedPVP_Score'] = gene_scores_weighted
                
    result_df = pd.DataFrame.from_dict(final_scores, orient='index')
    del result_df['allgens'] , result_df['varID'] #, result_df['CADD']
    result_df = result_df.sort_values(by='EmbedPVP_Score', ascending=False)
    result_df['Rank'] = result_df['EmbedPVP_Score'].rank(method='min', ascending=False)
    result_df = result_df.drop_duplicates()
    
    return result_df

def genomic_features(data_root, in_file, pathogenicity):
    gene_id = {}
    with open(f"{data_root}/Homo_sapiens.gene_info") as f:
        next(f)
        for line in f:
            d = line.split()
            Synonyms = d[4]
            Symbol = d[2]
            GeneID = d[1]
            alls = Synonyms.split('|')
            allg=[Symbol]
            for a in alls:
                allg.append(a)
            gene_id[GeneID] = allg
            
    data_cadd = pd.read_csv(pathogenicity, skiprows=2, compression='gzip', sep='\t', low_memory=False, names=['Chr','Start','Ref','Alt','RawScore','CADD'])                 
    
    data_cadd = data_cadd[['Chr','Start','Ref','Alt','CADD']]
    
    data = pd.read_csv(in_file, sep='\t', low_memory=False)
    data.rename(columns={'Otherinfo6':'ID'}, inplace=True)
    data['varID'] = data.reset_index().index
    data['Genes'] = data["Gene.knownGene"]  
    
    data = data.merge(data_cadd, on=['Chr','Start','Ref','Alt'], how='left')
    data['CADD'] = data['CADD'].fillna(0)   
    min_value = data['CADD'].min()
    max_value = data['CADD'].max()

    # Normalize the 'cadd_scores' column to the range [0, 1]
    data['CADD'] = (data['CADD'] - min_value) / (max_value - min_value)
            
    data = data[['Chr','ID','Start','Ref','Alt','Genes','Func.knownGene','GeneDetail.knownGene','varID','CADD']]

    data = data.T.to_dict()

    results = {}
    for k,v in data.items():
        results[k] = v
        dic = {}
        genes = str(v['Genes'])
        gene = genes.split(';')
        allgens = {}
        for i in gene:
            for agk,agv in gene_id.items():  
                if i in agv:
                    sgid = agk
                    break 
                else:
                    continue     
            n_sgid = str(sgid)
            allgens[n_sgid] = sgid
            
        if len(allgens) > 0 :
            results[k]['allgens'] = allgens
                     
    return results
    
if __name__ == '__main__':
    main()
