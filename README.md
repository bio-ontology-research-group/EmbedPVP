# EmbedPVP
Prioritizing genomic variants (SNP/InDls) through neuro-symbolic, knowledge-enhanced learning

## Annotation data sources (integrated in the candidate SNP prediction workflow)
We integrated the annotations from different sources:
- Gene ontology ([GO](http://geneontology.org/docs/download-go-annotations/))
- Mammalian Phenotype ontology ([MP](http://www.informatics.jax.org/vocab/mp_ontology))
- Human Phenotype Ontology ([HPO](https://hpo.jax.org/app/download/annotation))
- Uber-anatomy ontology ([UBERON](https://www.ebi.ac.uk/ols/ontologies/uberon))
- Gene expression in human Celltypes Ontology ([CL](https://www.nature.com/articles/s41586-018-0590-4))


## Dependencies

<div align="right">
<img src="https://raw.githubusercontent.com/bio-ontology-research-group/mowl/main/docs/source/mowl_black_background_colors_2048x2048px.png" alt="mOWL library" align="right" width="150">
</div>


- The code was developed and tested using python 3.7. To install python dependencies run:  
 `pip install -r requirements.txt`

- We used ([mOWL](https://github.com/bio-ontology-research-group/mowl)) library to process the input dataset as well as generated the embedding representation using different 
embedding-based approaches.



## Installation

```
pip install embedpvp
```

## Running EmbedPVP using pretrained models:
1. Download the distribution file and set up environment
```
git clone https://github.com/bio-ontology-research-group/EmbedPVP.git
cd EmbedPVP
conda env create -f environment.yml
conda activate embedpvp
```
2. Download all the files from [data]() and place the uncompressed the file in the folder named `/data`.
3. Download the required database using CADD by run:  `bash src/run_cadd.sh`
4. Run the command `embedpvp --help` to display help and parameters:

```
EmbedPVP: Prioritizing Causative Variants by Integrating Functional Embedding and Biological Annotations for Genes.

## Arguments:

- `-h, --help`: Show this help message and exit
- `-inputfile [INPUTFILE]`: Path to the VCF file
- `-ontology [OWL]`: Path to the ontology file in OWL format
- `-hpo [HPO]`: List of phenotype codes separated by commas
- `-model [MODEL]`: Preferred model (go, mp, uberon, hp, cl, union), default='hp'
- `-embedding [EMBED]`: Preferred embedding model (TransE, TransD, DL2vec), default='DL2vec'
- `-pathogenicity [PATH]`: Path to the pathogenicity prediction file (CADD)
- `-outfile [OUTFILE]`: Path to the results output file

```
### Run the example :

```
sh embedPVP \
    -inputfile data/example.vcf \
    -ontology data/upheno.owl \
    -hpo HP:0003701,HP:0001324,HP:0010628,HP:0003388,HP:0000774,HP:0002093,HP:0000508,HP:0000218,HP:0000007 \
    -model hp \
    -embedding DL2vec \
    -pathogenicity data/cadd_example.tsv \
    -outfile example_output.tsv   	

 Annotate VCF file (example.vcf) with the phenotypes (HP:0003701,HP:0001324,HP:0010628,HP:0003388,HP:0000774,HP:0002093,HP:0000508,HP:0000218,HP:0000007)...
 |========                        | 25% Annotated files generated successfully.
 |================                | 50% Phenotype prediction...
 |========================        | 75% SNP Prediction...
 |================================| 100%
The analysis is Done. You can find the priortize list in the output file: example_output.txt 

```

#### Output:
The script will output a ranking a score for the candidate caustive list of variants. 

## Note
For any questions or comments please contact azza.althagafi@kaust.edu.sa
