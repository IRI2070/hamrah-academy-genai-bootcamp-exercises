from sentence_transformers import CrossEncoder, SentenceTransformer, util
import numpy as np

with open('legal_names.txt', 'r') as f:
    registered_names = f.readlines()

bi_encoder = SentenceTransformer("IRI2070/tooka-sbert-large-v2-legal-names-bi-encoder", device="cuda")

cross_encoder = CrossEncoder("IRI2070/legal-names-validation-rules-classifier", device="cuda")

corpus_embeddings = bi_encoder.encode(registered_names, convert_to_tensor=True, show_progress_bar=True)


def search(query, top_k=10, threshold=0.70):
    print("candidate name:", query)

    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cuda()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]
    hits = [hit for hit in hits if hit['score'] >= threshold]

    cross_inp = [[query, registered_names[hit["corpus_id"]]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    probabilities = np.exp(cross_scores) / np.sum(np.exp(cross_scores), axis=1, keepdims=True)
    rules_mapping = ['abbreviation_shortening', 'activity_change', 'adjective_removal', 'domain_similarity',
                     'generic_word', 'minor_spelling_variations', 'morphological_variation', 'no_rule', 'prefix_suffix',
                     'singular_plural', 'synonym', 'word_order', 'word_removal']
    for idx in range(len(cross_scores)):
        _argmax = cross_scores[idx].argmax()
        hits[idx]["cross-score"] = probabilities[idx][_argmax]
        hits[idx]["rules"] = rules_mapping[_argmax]
        hits[idx]["name"] = registered_names[hits[idx]["corpus_id"]]
    print(hits)
