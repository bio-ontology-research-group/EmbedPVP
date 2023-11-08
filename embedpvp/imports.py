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
import tensorflow as tf
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
from ELEmModule import ELEmModule, ELEmbeddings, ELBoxEmbeddings
from mowl.projection.factory import projector_factory, PARSING_METHODS
from mowl.projection import DL2VecProjector, OWL2VecStarProjector
from mowl.projection import TaxonomyWithRelationsProjector
from mowl.nn import ELModule
from pykeen.triples import TriplesFactory
from gensim.models import KeyedVectors
from get_dis  import GDADataset 


