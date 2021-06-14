#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J DL2vec
#SBATCH -o logs/DL2vec.%J.out
#SBATCH -e logs/DL2vec.%J.err
#SBATCH --time=40:00:00
#SBATCH --mem=100G

module load gcc/6.4.0
module load groovy/3.0.6

#cp -r ../data /tmp/

onto_type=$1

onto=../data/phenomenetsh.owl
input=../data/${onto_type}_associations
output=../data/${onto_type}_embedding_model
entity_list=../data/${onto_type}_entity_List.pkl

echo $onto_type

python runDL2vec.py -embedsize 100 -ontology $onto  -associations $input  -outfile $output -entity_list $entity_list -num_workers 64 -file_prefix $onto_type

