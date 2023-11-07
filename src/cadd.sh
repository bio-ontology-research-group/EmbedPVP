#!/bin/bash
#SBATCH --job-name=cadd
#SBATCH --time=10:00:00
#SBATCH --mem=20G
#SBATCH --error=logs/cadd_%j.err
#SBATCH --output=logs/cadd_%j.out
#SBATCH --account=c2014

filename=$1
out=$2

module load bioperl/1.7.5/perl-5.30.0
module load perl 

module load cadd/1.6.1
module load vep
module load anaconda2

CADD.sh -g GRCh38 -p -q -o $out $filename
#CADD.sh -g GRCh37 -o $out $filename

