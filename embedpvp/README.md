# EmbedPVP
Prioritizing genomic variants (SNP/InDls) through neuro-symbolic, knowledge-enhanced learning

## Annotation data sources (integrated in the candidate SNP prediction workflow)
We integrated the annotations from different sources:
- Gene ontology ([GO](http://geneontology.org/docs/download-go-annotations/))
- Mammalian Phenotype ontology ([MP](http://www.informatics.jax.org/vocab/mp_ontology))
- Human Phenotype Ontology ([HPO](https://hpo.jax.org/app/download/annotation))
- Uber-anatomy ontology ([UBERON](https://www.ebi.ac.uk/ols/ontologies/uberon))

## Dependencies

<div align="right">
<img src="https://raw.githubusercontent.com/bio-ontology-research-group/mowl/main/docs/source/mowl_black_background_colors_2048x2048px.png" alt="mOWL library" align="right" width="150">
</div>


- The code was developed and tested using python 3.9. 

- We used ([mOWL](https://github.com/bio-ontology-research-group/mowl)) library to process the input dataset as well as generate the embedding representation using different 
embedding-based methods.

## Installation

```
pip install embedpvp
```

2. Download all the files from [data](https://drive.google.com/file/d/1QQVG_hzYl1X-rO64zgOX0xvoxv1Ux9of/view?usp=drive_link) and place the uncompressed the file in the folder named `/data`.
3. Download the required database using [CADD](https://cadd.gs.washington.edu/score).
4. Run the command `embedpvp --help` to display help and parameters:

```
EmbedPVP: Prioritizing Causative Variants by Integrating Functional Embedding and Biological Annotations for Genes.


Options:
  -d, --data-root TEXT      Data root folder  [required]
  -i, --in_file TEXT        Annotated Input VCF file  [required]
  -p, --pathogenicity TEXT  Path to the pathogenicity prediction file (CADD) [required]
  -hpo, --hpo TEXT          List of phenotype codes separated by commas [required]
  -m, --model_type TEXT     Ontology model, one of the following (go , mp , hp, uberon, union)
  -e, --embedding TEXT      Preferred embedding model (e.g. TransD, TransE, TranR, ConvE ,DistMult, DL2vec, OWL2vc, EL, ELBox)
  -dir, --outdir TEXT       Path to the output directory
  -o, --outfile TEXT        Path to the results output file
  --help                    Show this message and exit.

```

### Run the example:

```
embedpvp -d data/ -i example_annotation.vcf.hg38_multianno.txt  -p example_cadd.tsv.gz -hpo HP:0004791,HP:0002020,HP:0100580,HP:0001428,HP:0011459 -m hp -e TransE -dir output/ -o example_output1.tsv

 Annotate VCF file (example.vcf) with the phenotypes (HP:0003701,HP:0001324,HP:0010628,HP:0003388,HP:0000774,HP:0002093,HP:0000508,HP:0000218,HP:0000007)...
 |========                        | 25% Annotated files generated successfully.
 |================                | 50% Phenotype prediction...
 |========================        | 75% Variants prediction...
 |================================| 100%
The analysis is Done. You can find the priortize list in the output file: example_output.txt 

```

#### Output:
The script will output a ranking a score for the candidate caustive list of variants. 

## Note
For any questions or comments please contact azza.althagafi@kaust.edu.sa
