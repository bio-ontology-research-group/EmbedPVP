#!/usr/bin/env python
from imports import *

# Set random seed
np.random.seed(42)

# Configure logging
logging.getLogger("urllib3").setLevel(logging.WARNING)
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# Pandas options
pd.options.mode.chained_assignment = None  # default='warn'

# Threading lock
lock = Lock()

# Logger configuration
logger = logging.getLogger('my-logger')
logger.propagate = False
logger.disabled = True

# Temporary directory
tmp = tempfile.mkdtemp(prefix='EmbedPVP', suffix='/')

@ck.command()
@ck.option('--data-root', '-d', default='data/', help='Data root folder', required=True)
@ck.option('--in_file', '-i', help='Annotated Input VCF file', required=True)
@ck.option('--pathogenicity', '-p', help='Path to the pathogenicity prediction file (CADD)', required=True)
@ck.option('--hpo','-hpo', help='List of phenotype codes separated by commas', required=True)
@ck.option('--model_type','-m', help='Ontology model, one of the following (go , mp , hp, uberon, union)', default='hp')
@ck.option('--embedding','-e', help='Preferred embedding model (e.g. TransD, TransE, TranR, ConvE ,DistMult, DL2vec, OWL2vc, EL, ELBox)', default='dl2vec')
@ck.option('--outdir','-dir', default='output/', help='Path to the output directory')
@ck.option('--outfile','-o', default='embedpvp_results.tsv', help='Path to the results output file')

def main(data_root, in_file, pathogenicity, hpo, model_type, embedding, outdir, outfile):
    # Check data folder and required files
    try:
        if os.path.exists(data_root):
            in_file = os.path.join(data_root, in_file)
            owl_file = os.path.join(data_root, f'mowl_{model_type}.owl')
            pathogenicity_scores = os.path.join(data_root, pathogenicity)
            if not os.path.exists(in_file):
                raise Exception(f'Annotated input file ({in_file}) is missing!')
            if not os.path.exists(owl_file):
                raise Exception(f'Model file ({owl_file}) is missing!')
            if not os.path.exists(pathogenicity_scores):
                raise Exception(f'Pathogenicity prediction file ({pathogenicity_scores}) is missing!')
        else:
            raise Exception(f'Data folder ({data_root}) does not exist!')
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    # Initialize progress bar
    bar = Bar(max=4, fill='=', suffix='%(percent)d%%')

    state = 'START'
    while state != 'DONE':
        # Read input data, Load and Run pheno model
        bar.next()
        #ontology = mowl_insert_annotations(data_root, hpo, owl_file)
        print(f" Insert annotations to the ({model_type}) ontology...")

        if not os.path.exists(f'{outdir}/ontology.owl'):
            raise Exception(f'Ontology file with the inserted annotation not found ({outdir}/ontology.owl) is missing!')

        bar.next()
        # Call the pheno_models_embeddings function with appropriate arguments
        results = embedpvp_predictions(f'{outdir}/ontology.owl', embedding, in_file, pathogenicity)
        print(" Phenotype prediction...")

        out_file = os.path.join(outdir, outfile)
        results.to_csv(out_file, sep='\t', index=False)
        if not os.path.exists(out_file):
            raise Exception(f'The output file does not exist!')
        else:
            # Write the results to a file
            bar.next()
            print(f' DONE! You can find the prediction results in the output file: {outfile}')

        state = 'DONE'


def mowl_insert_annotations(data_root, hpo, owl_file, output_directory):
    """
    Insert annotations into an ontology.

    Args:
        data_root (str): Root directory for data.
        hpo (str): Comma-separated list of HPO terms.
        owl_file (str): Path to the OWL file.
        output_directory (str): Directory for output files.

    This function inserts annotations into the ontology specified by `owl_file`.
    """
    with open(f'{output_directory}/pheno.txt', "w") as fp:
        fp.write('patient' + "\t")
        for pheno in hpo.split(','):
            fp.write("http://purl.obolibrary.org/obo/" + pheno.replace(':', '_') + "\t")

    print("Reading the input phenotypes...")

    # Define annotations for ontology construction
    annotations = [(f'{output_directory}/pheno.txt', "http://has_annotation", True)]

    # Insert annotations into the ontology
    insert_annotations(owl_file, annotations, f'{output_directory}/ontology.owl')


def graph_based_embeddings(ds, method, outputdir, vector_size=100, window=5, epochs=1, num_walks=1, walk_length=30):
    """
    Generate graph-based embeddings using DeepWalk or similar methods.

    Args:
        ds: Your dataset or ontology object.
        method (str): The method to use ('dl2vec' or 'owl2vec').
        outputdir (str): The directory for saving output files.
        vector_size (int): The size of the word vectors.
        window (int): The size of the context window.
        epochs (int): The number of training epochs.
        num_walks (int): The number of walks per node.
        walk_length (int): The length of each walk.

    Returns:
        gensim.models.KeyedVectors: Word vectors.
    """
    walks_file = f'{outputdir}/{method}_walks.txt'

    if method == 'dl2vec':
        projector = DL2VecProjector(True)
        train_edges = projector.project(ds.ontology)
    elif method == 'owl2vec':
        projector = OWL2VecStarProjector(True)
        train_edges = projector.project(ds.ontology)

    walker = DeepWalk(num_walks=num_walks, walk_length=walk_length, restart_probability=0.1, workers=10, outfile=walks_file, seed=42)
    walks = walker.walk(train_edges)
    sentences = LineSentence(walks_file)

    model = Word2Vec(sentences, vector_size=vector_size, epochs=epochs, window=window, min_count=1, workers=10)
    word2vec_file = f'{outputdir}/{method}_embed.pkl'
    model.save(word2vec_file)

    vectors = model.wv
    return vectors


def semantic_embeddings(ds, method, outputdir, vector_size=100, margin=0.1, neg_norm=1.0, epochs=1, learning_rate=0.01, batch_size=100, device="cuda"):
    """
    Generate semantic embeddings using ELEmbeddings or ELBoxEmbeddings methods.

    Args:
        ds: Your dataset or ontology object.
        method (str): The method to use ('el' or 'elbox').
        outputdir (str): The directory for saving output files.
        vector_size (int): The size of the embedding vectors.
        margin (float): Margin value for loss function.
        neg_norm (float): Regularization norm for negative samples.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for training.
        batch_size (int): Batch size for training.
        device (str): Device for training (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple of numpy arrays: Entity and relation embeddings.
    """
    model_path = f"{outputdir}/{method}.pt"
    
    proj = DL2VecProjector(True)
    
    if method == 'el':
        model = ELEmbeddings(ds,
            embed_dim=vector_size,
            margin=margin,
            reg_norm=neg_norm,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            model_filepath=model_path,
            device=device)  

    elif method == 'elbox':
        model = ELBoxEmbeddings(ds,
            embed_dim=vector_size,
            margin=margin,
            reg_norm=neg_norm,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            model_filepath=model_path,
            device=device)
                                      
    model.train()
    model.load_best_model()
    ent_embs, rel_embs = model.get_embeddings()

    with open(f'{outputdir}/{method}_embed.pkl', "wb") as f:
        pkl.dump(ent_embs, f)
    with open(f'{outputdir}/{method}_rel.pkl', "wb") as f:
        pkl.dump(rel_embs, f)
        
    return ent_embs, rel_embs


def translation_embeddings(ds, method, outputdir, vector_size=100, epochs=1, learning_rate=0.01, batch_size=100, device="cuda"):
    """
    Generate translation-based embeddings using various models.

    Args:
        ds: Your dataset or ontology object.
        method (str): The method to use ('TransD', 'ConvE', 'DistMult', 'TransR', 'TransE').
        outputdir (str): The directory for saving output files.
        vector_size (int): The size of the embedding vectors.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for training.
        batch_size (int): Batch size for training.
        device (str): Device for training (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple of dictionaries: Entity and relation embeddings.
    """
    proj = DL2VecProjector(True)
    train_edges = proj.project(ds.ontology)
    model_path = f"{outputdir}/{method}.pt"

    # Extract unique classes and relations
    classes, relations = Edge.getEntitiesAndRelations(train_edges)
    classes, relations = list(set(classes)), list(set(relations))
    classes.sort()
    relations.sort()
    class_to_id = {c: i for i, c in enumerate(classes)}
    rel_to_id = {r: i for i, r in enumerate(relations)}

    # Create a PyKEEN TriplesFactory
    triples_factory = TriplesFactory.from_labeled_triples(
        labeled_triples=train_edges,
        entity_to_id=class_to_id,
        relation_to_id=rel_to_id,
    )

    # Initialize the PyKEEN model based on the selected method
    if method == 'TransD':
        pk_model = TransD(
            triples_factory=triples_factory,
            embedding_dim=vector_size,
            random_seed=42,
        )
    elif method == 'ConvE':
        pk_model = ConvE(
            triples_factory=triples_factory,
            embedding_dim=vector_size,
            random_seed=42,
        )
    elif method == 'DistMult':
        pk_model = DistMult(
            triples_factory=triples_factory,
            embedding_dim=vector_size,
            random_seed=42,
        )
    elif method == 'TransR':
        pk_model = TransR(
            triples_factory=triples_factory,
            embedding_dim=vector_size,
            random_seed=42,
        )
    elif method == 'TransE':
        pk_model = TransE(
            triples_factory=triples_factory,
            embedding_dim=vector_size,
            random_seed=42,
        )

    # Create a KGEModel and train the model
    model = KGEModel(
        triples_factory=triples_factory,
        model=pk_model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        model_filepath=model_path,
    )
    model.train()
    model.load_best_model()
    ent_embs = model.class_embeddings_dict
    rel_embs = model.object_property_embeddings_dict

    # Save embeddings to files
    with open(f'{outputdir}/{method}_embed.pkl', "wb") as f:
        pkl.dump(ent_embs, f)

    with open(f'{outputdir}/{method}_rel.pkl', "wb") as f:
        pkl.dump(rel_embs, f)

    return ent_embs, rel_embs

                
def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def calculate_graph_embed_similarity(variants_features, embeddings, weight=0.6):
    final_scores = {}
    
    for variant_id, variant_data in features_dict.items():
        cadd_score = variant_data['features'][0]
        gene_scores = [cadd_score]

        for gene, gene_embedding in variant_data['genes'].items():
            patient_gene_key = f'{patient_id}_{gene}'
            if patient_gene_key in gene_predictions:
                gene_score = gene_predictions[patient_gene_key]
                gene_scores.append(gene_score)
        
        max_gene_score = max(gene_scores)
        gene_scores_weighted = (weight * max_gene_score) + ((1 - weight) * cadd_score)
        final_scores[variant_data['varID']] = gene_scores_weighted

    result_df = pd.DataFrame(final_scores.items(), columns=['ID', 'Score'])
    result_df = result_df.sort_values(by='Score', ascending=False)
    result_df['Rank'] = result_df['Score'].rank(method='min', ascending=False)
    results = result_df.groupby(['ID'])['Rank'].min().reset_index()

    return results
    
def calculate_semantic_embed_similarity(variants_features, entity_embedding, relation_embedding, weight=0.6):
    final_scores = {}
    predictions = {}

    # Calculate predictions based on entity embeddings
    for background_gene in all_genes:
        disease_entity = 'http://ontology.com/' + ID_patient
        gene = background_gene.split('/')[-1]
        relation = 'http://is_associated_with'
        feature_key = ID_patient + '_' + gene

        if gene in class_to_id and disease_entity in class_to_id and relation in rel_to_id:
            disease_entity_id = class_to_id[disease_entity]
            gene_id = class_to_id[gene]
            relation_id = rel_to_id[relation]
            triple = (gene_id, relation_id, disease_entity_id)
            tensor_triple = th.tensor(triple).reshape(1, 3)
            score = model.model.gci2_loss(tensor_triple)
            score = score.item() / 100
            predictions[feature_key] = score

    for key, value in dic.items():
        ID = value['varID']
        cadd = value['features']
        all_scores = []
        all_scores.append(cadd[0])
        similarities = []
        score = 0

        for gene_id, gene_embedding in value['genes'].items():
            gene_feature_key = ID_patient + '_' + str(gene_id)

            if gene_feature_key in predictions:
                score = predictions[gene_feature_key]

            similarities.append(score)

        max_similarity = max(similarities)
        all_scores.append(max_similarity)
        final_score = (weight * all_scores[1]) + ((1 - weight) * all_scores[0])
        final_scores[ID] = final_score

    result_df = pd.DataFrame(final_scores.items(), columns=['ID', 'Score'])
    result_df = result_df.sort_values(by='Score', ascending=False)
    result_df['Rank'] = result_df['Score'].rank(method='min', ascending=False)
    results = result_df.groupby(['ID'])['Rank'].min().reset_index()

    return results

def calculate_translation_embed_similarity(variants_features, entity_embedding, relation_embedding, weight=0.6):
    final_scores = {}
    for key, value in dic.items():
        ID = value['varID']
        cadd_score = value['features'][0]
        all_scores = []
        all_scores.append(cadd_score)
        similarities = []

        for gene_id, gene_embedding in value['genes'].items():
            gene = gene_id
            gene_url = 'http://ontology.com/' + str(gene)
            disease_url = 'http://ontology.com/' + ID_patient
            relation_url = 'http://is_associated_with'

            if gene_url in class_to_id and disease_url in class_to_id and relation_url in rel_to_id:
                disease_entity_id = class_to_id[disease_url]
                gene_id = class_to_id[gene_url]
                relation_id = rel_to_id[relation_url]
                triple = (gene_id, relation_id, disease_entity_id)
                score = model.score_method_point(triple).item()
            else:
                score = 0
                
            similarities.append(score)

        max_similarity = max(similarities)
        all_scores.append(max_similarity)
        final_score = (weight * all_scores[1]) + ((1 - weight) * all_scores[0])
        final_scores[ID] = final_score

    result_df = pd.DataFrame(final_scores.items(), columns=['ID', 'Score'])
    result_df = result_df.sort_values(by='Score', ascending=False)
    result_df['Rank'] = result_df['Score'].rank(method='min', ascending=False)
    results = result_df.groupby(['ID'])['Rank'].min().reset_index()

    return results

#def genomic_features(in_file, pathogenicity):
    
    

def embedpvp_predictions(ontology, embedding_type, in_file, pathogenicity):
    ds = PathDataset(ontology, validation_path=None, testing_path=None)
    variants_features = genomic_features(in_file, pathogenicity)
    
    embedding_type = embedding_type.lower()

    if embedding_type in ['dl2vec', 'owl2vec']:
        embeddings = graph_based_embeddings(ds, embedding_type)
        bar.next()
        ranks = calculate_graph_embed_similarity(variants_features, embeddings)

    elif embedding_type in ['el', 'elbox']:
        ent_embedding, rel_embedding = semantic_embeddings(ds, embedding_type)
        bar.next()
        ranks = calculate_semantic_model_similarity(variants_features, ent_embedding, rel_embedding)

    elif embedding_type in ['transd', 'transe', 'tranr', 'conve', 'distmult']:
        ent_embedding, rel_embedding = translation_embeddings(ds, embedding_type)
        bar.next()
        ranks = calculate_translation_similarity(variants_features, ent_embedding, rel_embedding)

    print(" Variants prediction...")
    return ranks

    
if __name__ == '__main__':
    main()