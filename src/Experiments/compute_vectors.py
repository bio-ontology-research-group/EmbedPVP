from __future__ import print_function
import numpy as np
import random
import json
import sys
import os
import gensim

import networkx as nx
from networkx.readwrite import json_graph
import multiprocessing as mp
from threading import Lock
import pickle as pkl


lock = Lock()
WALK_LEN=30
N_WALKS=100
global data_pairs
data_pairs = []

def run_random_walks(G, nodes, num_walks=N_WALKS, file_prefix=''):
    print("now we start random walk")
    pairs = []
    for count, node in enumerate(nodes):

        if G.degree(node) == 0:
            continue

        epsilon = 0.000001
        for i in range(num_walks):
            curr_node = node
            walk_accumulate=[]
            for j in range(WALK_LEN):
                neighbours = list(G.neighbors(curr_node))
                neighbour_types = [G.edges[curr_node, neighbour]['type'] for neighbour in neighbours]
                num_HasAssociations = neighbour_types.count('HasAssociation')

                # make HasAssociation and non-HasAssociation equally likely
                HasAssociation_weight = 0.01 / (num_HasAssociations + epsilon)
                non_HasAssociation_weight = 0.99 / (len(neighbours)-num_HasAssociations + epsilon)

                # build weight vector
                weight_vec = [(HasAssociation_weight if type=='HasAssociation' else non_HasAssociation_weight) for type in neighbour_types]

                # next_node = random.choice(list(G.neighbors(curr_node)))
                next_node = random.choices(population=neighbours, weights=weight_vec, k=1)[0]

                type_nodes = G.edges[curr_node, next_node]["type"]

                if curr_node ==node:
                    walk_accumulate.append(curr_node)
                walk_accumulate.append(type_nodes)
                walk_accumulate.append(next_node)

                curr_node = next_node

            pairs.append(walk_accumulate)
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    write_file(pairs, file_prefix)


def run_walk(nodes,G, num_workers=48, file_prefix=''):
    global data_pairs
    number = num_workers
    length = len(nodes) // number

    processes = [mp.Process(target=run_random_walks, args=(G, nodes[(index) * length:(index + 1) * length], N_WALKS, file_prefix)) for index
                 in range(number-1)]
    processes.append(mp.Process(target=run_random_walks, args=(G, nodes[(number-1) * length:len(nodes) - 1], N_WALKS, file_prefix)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def write_file(pair, file_prefix=''):
    with lock:
        with open(file_prefix + "_walks.txt", "a") as fp:
            for p in pair:
                for sub_p in p:
                    fp.write(str(sub_p)+" ")
                fp.write("\n")


def gene_node_vector(graph, entity_list, outfile, embedding_size, file_prefix='', num_workers=48):
    nodes_set=set()
    #with open(entity_list,"r") as f:
    #    for line in f.readlines():
    #        data = line.strip().split()
    #        for da in data:
    #            nodes_set.add(da)

    with open(entity_list,"rb") as f:
        nodes_set= pkl.load(f)
            
    nodes_G = [n for n in graph.nodes()]

    G = graph.subgraph(nodes_G)
    nodes= [n for n in nodes_set]
    run_walk(nodes,G, num_workers=num_workers, file_prefix=file_prefix)

    print("start to train the word2vec models")
    sentences=gensim.models.word2vec.LineSentence(file_prefix+"_walks.txt")
    model=gensim.models.Word2Vec(sentences,sg=1, min_count=1, size=embedding_size, window=10,iter=30,workers=num_workers)
    model.save(outfile)
