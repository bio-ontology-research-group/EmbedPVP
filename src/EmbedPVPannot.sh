#!/bin/bash

# Script to annotate the input VCF file 
# To run: bash EmbedPVPannot.sh input_file.vcf

path_to_vcf=$1 
path_to_output="$1"_annovar
path_to_annovar=""

perl "$path_to_annovar"/table_annovar.pl $path_to_vcf \
     "$path_to_annovar"/humandb/ -buildver hg38 -out "$path_to_output" \
     -remove -protocol refGene,dbnsfp30a,mcap,dbscsnv11,clinvar_20190305,revel \
     -operation g,f,f,f,f,f -nastring . -vcfinput
     
[ -e "$path_to_output"/*.vcf ] && rm "$path_to_output"/*.vcf
[ -e "$path_to_output"/*.avinput ] && rm  "$path_to_output"/*.avinput

