#!/usr/bin/env python
import click as ck
import numpy as np
import pandas as pd
import gzip
import os
import sys
import logging
import numpy
import tensorflow as tf
import gensim
import random
import pickle as pkl
from pandas.api.types import CategoricalDtype
from statistics import mean
import sklearn.metrics as metrics
from progress.bar import Bar
import time
import subprocess
import argparse
import os.path
import networkx as nx
import pickle as pkl
import json
import tempfile
import shutil
import pdb
from os import path
import networkx as nx
import json
import multiprocessing as mp
from threading import Lock
import torch
import shutil
from torch._C import *
import torch.optim as optim
from scipy.stats import rankdata
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from networkx.readwrite import json_graph
from gensim.models import Word2Vec, Phrases, phrases, KeyedVectors
from random import seed
logging.getLogger("urllib3").setLevel(logging.WARNING)
np.random.seed(42)
from statistics import mean
from operator import itemgetter
import scipy.stats as ss
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tmp = tempfile.mkdtemp(prefix='DeepSVP', suffix='/')
bar = Bar(max=4, fill='=', suffix='%(percent)d%%')
lock = Lock()
WALK_LEN = 25
N_WALKS = 100

logger = logging.getLogger('my-logger')
logger.propagate = False
logger.disabled = True

@ck.command()
@ck.option('--data-root', '-d',
           default='data/',
           help='Data root folder',
           required=True)
@ck.option('--in-file', '-i', help='Annotated Input file', required=True)
@ck.option('--hpo','-p',
           help='List of phenotype ids separated by commas',
           required=True)
@ck.option('--maf_filter','-maf',
           help='Allele frequency filter using gnomAD and 1000G default<=0.01',
           default=0.01)
@ck.option('--model_type','-m',
            default='mp',
            help='Ontology model, one of the following (go , mp , hp, cl, uberon, union), default=mp')
@ck.option('--aggregation','-ag',
            help='Aggregation method for the genes within CNV (max or mean) default=max',
            default='max')
@ck.option('--outfile','-o',
            default='cnv_results.tsv',
            help='Output result file')
def main(data_root, in_file, hpo, maf_filter, model_type, aggregation, outfile):
    # Check data folder and required files
    """DeepSVP: A phenotype-based tool to prioritize caustive CNV using WGS data and Phenotype/Gene Functional Similarity"""
    try:
        if os.path.exists(data_root):
            in_file = os.path.join(data_root, in_file)
            model_file = model_type + '_' + aggregation + '.h5'
            model_file = os.path.join(data_root, model_file)
            if not os.path.exists(in_file):
                raise Exception(
                    f'Annotated Input file ({in_file}) is missing!')
            if not os.path.exists(model_file):
                raise Exception(f'Model file ({model_file}) is missing!')
        else:
            raise Exception(f'Data folder {data_root} does not exist!')
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    state = 'START'
    while state != 'DONE':
        # Read input data, Load and Run pheno model
        bar.next()
        load_pheno_model(in_file, hpo, model_type, data_root)

        bar.next()
        print(" Phenotype prediction... ")

        # Load and Run cnv model
        bar.next()
        output = load_cnv_model(in_file, model_type, aggregation, data_root, maf_filter)
        print(" CNV Prediction... ")

        # Write the results to a file
        bar.next()
        print_results(output, outfile)
        print(' DONE! You can find the prediction results in the output file:',
              outfile)

        state = 'DONE'

