import sys
sys.path.append('../../../')
import os
import random
from math import floor
import pandas as pd
import logging
import numpy as np

root = "data/"
root_mouse = "data_mouse/"
root_human = "data_human/"
root_go = "data_go/"
root_hp = "data_hp/"
root_uberon = "data_uberon/"
root_union = "data_union/"

def gene_union_annots(str1, str2, str3, str4, verbose = False):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    out_file = root + 'union_annots.tsv'
    df1_go = pd.read_csv(str1, header=None)
    df2_mp = pd.read_csv(str2, header=None)
    df3_hp = pd.read_csv(str3, header=None)
    df4_uberon = pd.read_csv(str4, header=None)
    
    res = pd.concat([df1_go, df2_mp, df3_hp, df4_uberon], ignore_index=True).drop_duplicates(0)
    res.to_csv(out_file, header=False, index=False)

def gene_hp_annots(verbose = False):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    in_file = root + 'phenotype.hpoa'
    out_file = root + 'hp_annots.tsv'

    diseases = pd.read_csv(in_file, skiprows=4, sep='\t', low_memory=False)
    diseases = diseases[['#DatabaseID','HPO_ID']]
    diseases.columns = ['OMIM IDs','HPO_ID']

    #DO Disease ID   DO Disease Name OMIM IDs        Common Organism Name    NCBI Taxon ID   Symbol  EntrezGene ID   Mouse MGI ID
    ##DatabaseID     DiseaseName     Qualifier       HPO_ID  Reference       Evidence        Onset   Frequency       Sex     Modifier        Aspect  Biocuration

    diseases_genes = pd.read_csv('data/MGI_DO.rpt', sep='\t', low_memory=False)
    diseases_genes = diseases_genes[['OMIM IDs','EntrezGene ID']]
    
    diseases_genes = diseases_genes[diseases_genes['EntrezGene ID'].notna()]
    diseases_genes['EntrezGene ID'] = diseases_genes['EntrezGene ID'].astype(int)
    
    genes_hpo = diseases_genes.merge(diseases, on='OMIM IDs')
    genes_hpo = genes_hpo.drop_duplicates()
    del genes_hpo['OMIM IDs']

    genes_hpo = genes_hpo.groupby('EntrezGene ID')['HPO_ID'].apply(list).to_dict()

    with open(out_file, 'w') as fout:
        for gene, pheno in genes_hpo.items():
            id_gene = str(gene)
            phen_annots = list(map(lambda x: "http://purl.obolibrary.org/obo/" + x.replace(":", "_"), pheno))
            out = "\t".join([id_gene] + phen_annots)
            fout.write(out+"\n")

def gene_uberon_annots(verbose = False):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    in_file = root + 'E-MTAB-5214-query-results.tpms.tsv'
    out_file = root + 'uberon_annots.tsv'

    # obtain the gene and uberon associations
    gene_expression=pd.read_table(in_file)
    gene_expression=gene_expression.drop(columns="EBV-transformed lymphocyte")
    gene_expression=gene_expression.drop(columns="transformed skin fibroblast")

    anatomy_uberon={'Brodmann (1909) area 24': 'UBERON_0006101',
     'Brodmann (1909) area 9': 'UBERON_0013540',
     'C1 segment of cervical spinal cord': 'UBERON_0006469',
     'adrenal gland': 'UBERON_0002369',
     'amygdala': 'UBERON_0001876',
     'aorta': 'UBERON_0000947',
     'atrium auricular region': 'UBERON_0006618',
     'breast': 'UBERON_0000310',
     'caudate nucleus': 'UBERON_0001873',
     'cerebellar hemisphere': 'UBERON_0002245',
     'cerebral cortex': 'UBERON_0000956',
     'coronary artery': 'UBERON_0001621',
     'cortex of kidney': 'UBERON_0001225',
     'ectocervix': 'UBERON_0012249',
     'endocervix': 'UBERON_0000458',
     'esophagogastric junction': 'UBERON_0007650',
     'esophagus mucosa': 'UBERON_0002469',
     'esophagus muscularis mucosa': 'UBERON_0004648',
     'fallopian tube': 'UBERON_0003889',
     'greater omentum': 'UBERON_0005448',
     'heart left ventricle': 'UBERON_0002084',
     'hypothalamus': 'UBERON_0001898',
     'lower leg skin': 'UBERON_0004264',
     'minor salivary gland': 'UBERON_0001830',
     'nucleus accumbens': 'UBERON_0001882',
     'ovary': 'UBERON_0000992',
     'pancreas': 'UBERON_0001264',
     'pituitary gland': 'UBERON_0000007',
     'prostate gland': 'UBERON_0002367',
     'putamen': 'UBERON_0001874',
     'sigmoid colon': 'UBERON_0001159',
     "small intestine Peyer's patch": 'UBERON_0003454',
     'stomach': 'UBERON_0000945',
     'subcutaneous adipose tissue': 'UBERON_0002190',
     'substantia nigra': 'UBERON_0002038',
     'suprapubic skin': 'UBERON_0036149',
     'testis': 'UBERON_0000473',
     'thyroid gland': 'UBERON_0002046',
     'tibial artery': 'UBERON_0007610',
     'tibial nerve': 'UBERON_0001323',
     'transverse colon': 'UBERON_0001157',
     'urinary bladder': 'UBERON_0001255',
     'uterus': 'UBERON_0000995',
     'vagina': 'UBERON_0000996',
      'blood':"UBERON_0000178",
      'liver':"UBERON_0002107",
      'lung':"UBERON_0002048",
      "spleen":"UBERON_0002106",
      "cerebellum":"UBERON_0002037",
      "skeletal muscle tissue":"UBERON_0001134",
      "hippocampus proper":"UBERON_0002305"}
      
    # convert the nan value into 0 value
    for index in gene_expression.index:
      for name in gene_expression.columns[2:]:
          if (str(gene_expression.loc[index,name])=="nan"):
              gene_expression.loc[index,name]=0


    # now filter the uberon that has expression value more than 4.0

    gene_uberon_feature=dict()
    threshold=4.0
    ensembl_dic = dict()
    
    with open("data/gene2ensembl","r") as f:
        for line in f.readlines():
            line = line.split("\t")
            ensembl_dic[line[2]] = line[1]

    for index in gene_expression.index:
        name = gene_expression.loc[index,"Gene ID"]
        if name in ensembl_dic.keys():
            name = ensembl_dic[name]
            for column in gene_expression.columns[2:]:
                if gene_expression.loc[index,column]>=threshold:
                    try:                       
                        gene_uberon_feature[name].add(anatomy_uberon[column])
                    except:
                        temp_set=set()
                        temp_set.add(anatomy_uberon[column])
                        gene_uberon_feature[name]=temp_set

  
    
    print("Number of uberon   ",str(len(gene_uberon_feature)))
    with open(out_file, 'w') as fout:
        for gene, pheno in gene_uberon_feature.items():
            id_gene =  gene
            #phen_annots = pheno.split(", ")
            phen_annots = list(map(lambda x: "http://purl.obolibrary.org/obo/" + x.replace(":", "_"), pheno))
            out = "\t".join([id_gene] + phen_annots)
            fout.write(out+"\n")    
      

def gene_go_annots(verbose = False):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    in_file = root + 'goa_human.gaf'
    out_file = root + 'go_annots.tsv'

    human_to_mouse=dict()
    mouse_to_human=dict()
    geneName_to_id=dict()
    with open("data/HMD_HumanPhenotype.rpt",'r') as f:
        for line in f.readlines():
            data=line.split("\t")
            gene_name=data[0].strip()
            human_id=data[1].strip()
            mouse_id=data[3].strip()
            human_to_mouse[human_id]=mouse_id
            mouse_to_human[mouse_id]=human_id
            geneName_to_id[gene_name]=human_id

    #A1BG    1       A1bg    MGI:2152878  
    accession2gene=dict()
    with open("data/gene2accession","r") as f:
        for line in f.readlines():
            line = line.split("\t")
            if line[2]!= "PROVISIONAL":
                accession2gene[line[5][:-2]]=line[1]
    
    gene_go_feature=dict()
    gene_expression_name=set()
    with open(in_file,"r") as f:
        for line in f.readlines():
            data=line.split("\t")
            gene_id = data[1].strip()
    
            evidence_score=data[6].strip()
            go_id=data[4].strip()
    
            if not ((evidence_score=="IEA") or (evidence_score=="ND")):
                if gene_id in accession2gene.keys():
                    human_gene=accession2gene[gene_id]
                    try:
                        gene_go_feature[human_gene].append(go_id)
                    except:
                        gene_go_feature[human_gene]=[go_id]


    print("Number of go: ",str(len(gene_go_feature)))
    with open(out_file, 'w') as fout:
        for gene, pheno in gene_go_feature.items():
            id_gene =  gene
            #phen_annots = pheno.split(", ")
            phen_annots = list(map(lambda x: "http://purl.obolibrary.org/obo/" + x.replace(":", "_"), pheno))
            out = "\t".join([id_gene] + phen_annots)
            fout.write(out+"\n")

def gene_phen_annots(verbose = False):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    in_file = root + 'HMD_HumanPhenotype.rpt'
    out_file = root + 'mp_annots.tsv'
    all_gene=[]
    
    with open(out_file, 'w+') as fout:
        with open(in_file, 'r') as fin:
            for line in fin:
                line = line.strip().split('\t')
                if len(line) < 5:
                    continue
                h_gene, id_gene, m_gene, mgi, phen_annots = tuple(line)
                phen_annots = phen_annots.split(", ")

                id_gene = id_gene
                
                all_gene.append(id_gene)
                phen_annots = list(map(lambda x: "http://purl.obolibrary.org/obo/" + x.replace(":", "_"), phen_annots))
                out = "\t".join([id_gene] + phen_annots)
                fout.write(out+"\n")


def disease_phen_annots(verbose = False):
    if verbose:
        logging.basicConfig(level=logging.INFO)

    in_file = root + 'phenotype.hpoa'
    out_file = root + 'disease_annots.tsv'

    with open(out_file, 'w') as fout:
        with open(in_file, 'r') as fin:
            for line in fin:
                if line.startswith("#"):
                    continue
                
                line = line.strip().split('\t')

                disease_id = line[0]
                phenotype = line[3]

                out = disease_id + "\t" + "http://purl.obolibrary.org/obo/"+phenotype.replace(":", "_")
                fout.write(out+"\n")

def gene_disease_assoc(species, root_sp):

    in_file = root + "MGI_DO.rpt"
    out_file = root + f"gene_disease_assoc_{species}.tsv"

    with open(out_file, "w") as fout:
        with open(in_file, "r") as fin:
            for line in fin:
                if line.startswith("DO "):
                    continue

                line = line.strip().split('\t')

                if not species in line[3]:
                    continue
                disease_id = line[2]
                gene_id = line[6]

                if gene_id != "" and disease_id != "":
                    diseases = disease_id.split("|")
                    for disease in diseases:
                        out = gene_id + "\t" + disease 
                        fout.write(out+"\n")
                

def split_associations(in_file, species, root_sp):

    train_file = root_sp + f"train_assoc_data_{species}.tsv"
    valid_file = root_sp + f"valid_assoc_data_{species}.tsv"
    test_file = root_sp + f"test_assoc_data_{species}.tsv"

    with open(in_file, "r") as fin:
        assocs = fin.readlines()

    random.shuffle(assocs)

    n_assocs = len(assocs)
    train_idx = floor(n_assocs*0.8)
    valid_idx = train_idx + floor(n_assocs*0.1)
    test_idx = valid_idx + floor(n_assocs*0.1)

    train_assocs = assocs[:train_idx]
    valid_assocs = assocs[train_idx:valid_idx]
    test_assocs = assocs[valid_idx:]

    with open(train_file, "w") as f:
        for line in train_assocs:
            f.write(line)

    with open(valid_file, "w") as f:
        for line in valid_assocs:
            f.write(line)

    with open(test_file, "w") as f:
        for line in test_assocs:
            f.write(line)


def create_dir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
                        
if __name__ == "__main__":

    create_dir(root)
    create_dir(root_human)
    create_dir(root_mouse)
    create_dir(root_go)
    create_dir(root_hp)
    create_dir(root_uberon)
    create_dir(root_union)

    species = ["human", "mouse"] #["mouse", "human", "go", "hp", "uberon", "union"]

    for sp in species:
        if sp == "mouse":
            gene_phen_annots()
            print('mp done')
            root_sp = root_mouse
            in_file = root_sp + f"gene_disease_assoc_{sp}.tsv"
            gene_disease_assoc(sp, root_sp)
            split_associations(in_file, sp, root_sp)
                        
        elif sp == "human":
            disease_phen_annots()
            print('dis done')            
            root_sp = root_human
            in_file = root_sp + f"gene_disease_assoc_{sp}.tsv"
            gene_disease_assoc(sp, root_sp)
            split_associations(in_file, sp, root_sp)
                        
        elif sp == "go":
            gene_go_annots()
            print('go done')
            root_sp = root_go
        elif sp == "uberon":
            gene_uberon_annots()
            print('uberon done')
            root_sp = root_uberon
        elif sp == "hp":
            gene_hp_annots()
            print('hp done')
            root_sp = root_hp
        elif sp == "union":
            gene_union_annots(root+'go_annots.tsv',root+'mp_annots.tsv', root+'hp_annots.tsv', root+'uberon_annots.tsv')
            print('union done')
            root_sp = root_union
                    
        else:
            raise ValueError("Species name not recognized")

    
        
