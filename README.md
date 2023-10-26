# EmbedPVP
Prioritizing genomic variants (SNP/InDls) through neuro-symbolic, knowledge-enhanced learning

We developed an Embedding-based Phenotype Variant
  Predictor (EmbedPVP), a computational method to prioritize variants
  involved in genetic diseases by combining genomic information and
  clinical phenotypes. EmbedPVP leverages a large amount of background
  knowledge from human and model organisms about molecular mechanisms
  through which abnormal phenotypes may arise. Specifically, EmbedPVP
  incorporates phenotypes linked to genes, functions of gene products,
  and the anatomical site of gene expression, and systematically
  relates them to their phenotypic effects through neuro-symbolic,
  knowledge-enhanced machine learning.

  
## Annotation data sources (integrated in the candidate SNP prediction workflow)
We integrated the annotations from different sources:
- Gene ontology ([GO](http://geneontology.org/docs/download-go-annotations/))
- Mammalian Phenotype ontology ([MP](http://www.informatics.jax.org/vocab/mp_ontology))
- Human Phenotype Ontology ([HPO](https://hpo.jax.org/app/download/annotation))
- Uber-anatomy ontology ([UBERON](https://www.ebi.ac.uk/ols/ontologies/uberon))

As these annotations are available for different numbers
of genes, we also used the phenotypes based on the union of all genes
and their annotations (i.e., for genes that have annotations from one,
two or all four datasets, HPO, MP, GO, and Uberon). We used the
integrated phenotype ontology [uPheno](https://zenodo.org/records/3710690) as our
phenotype ontology to add background knowledge from biomedical
ontologies, as it integrates human and model organism phenotypes and
allows them to be compared.

## Dependencies

<div align="right">
<img src="https://raw.githubusercontent.com/bio-ontology-research-group/mowl/main/docs/source/mowl_black_background_colors_2048x2048px.png" alt="mOWL library" align="right" width="150">
</div>


- The code was developed and tested using python 3.7. To install python dependencies run:  
 `pip install -r requirements.txt`

- We used ([mOWL](https://github.com/bio-ontology-research-group/mowl)) library to process the input dataset as well as generated the embedding representation using different 
embedding-based methods.


## Note
For any questions or comments please contact azza.althagafi@kaust.edu.sa
