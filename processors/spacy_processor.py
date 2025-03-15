import spacy
from pydantic import BaseModel
from typing import List

# Load the spaCy model (ensure you have run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# ---------------------------
# Pydantic models for output (unchanged)
# ---------------------------
class Single(BaseModel):
    """
    Represents a single relationship between two entities.
    """
    node: str
    target_node: str
    relationship: str

class GraphComponents(BaseModel):
    """
    Represents the collection of relationships extracted from text.
    """
    graph: List[Single]

# ---------------------------
# Optional: Add an EntityRuler with custom patterns if needed.
# ---------------------------
ruler = nlp.add_pipe("entity_ruler", before="ner")
custom_patterns = [
    {"label": "ORG", "pattern": [{"LOWER": "neo4j"}]},  # Always label "Neo4j" as ORG
    {"label": "TECH", "pattern": "Python"}              # Label "Python" as TECH
    # You can add more patterns here as needed.
]
ruler.add_patterns(custom_patterns)

# ---------------------------
# Optional: Try to add coreference resolution if installed.
# ---------------------------
try:
    import coreferee
    nlp.add_pipe("coreferee", after="parser")
except ImportError:
    # Coreference resolution not available; continue without it.
    pass

# ---------------------------
# Helper function: Retrieve the full entity span for a token.
# ---------------------------
def get_entity_for_token(token, doc):
    """
    Returns the entity span covering the token if available.
    """
    for ent in doc.ents:
        if token.i >= ent.start and token.i < ent.end:
            return ent
    return None

# ---------------------------
# Main Extraction Function
# ---------------------------
def spacy_llm_parser(text: str) -> GraphComponents:
    """
    Extracts graph components using advanced dependency parsing and heuristics.
    
    This function processes the text sentence by sentence. It:
      - Uses dependency parsing to extract subject-verb-object relationships.
      - Leverages full entity spans from spaCy's NER.
      - Optionally uses coreference resolution (if available) to replace pronouns.
      - Falls back to consecutive entity linking if no dependency-based relationships are found.
    
    Returns:
        A GraphComponents object containing a list of Single relationships.
    """
    doc = nlp(text)
    graph_list = []

    # Process each sentence
    for sent in doc.sents:
        sent_text = sent.text
        # If coreference resolution is available, adjust sentence text (simplified example)
        if getattr(doc._, "coref_clusters", None):
            for cluster in doc._.coref_clusters:
                for mention in cluster:
                    # Replace simple pronouns with the cluster's main mention if within the sentence
                    if mention.text.lower() in ["he", "she", "they", "it", "him", "her", "them"]:
                        if mention.start >= sent.start and mention.end <= sent.end:
                            sent_text = sent_text.replace(mention.text, cluster.main.text)
            # Re-parse the adjusted sentence
            sent = nlp(sent_text)

        # Use dependency parsing to extract relationships
        for token in sent:
            if token.pos_ in ("VERB", "AUX"):
                # Find subjects (nsubj, nsubjpass) and objects (dobj, pobj, attr, oprd)
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr", "oprd")]
                if subjects and objects:
                    for subj in subjects:
                        subj_text = subj.text
                        # If token is part of a larger entity, get the full span
                        ent_subj = get_entity_for_token(subj, doc)
                        if ent_subj:
                            subj_text = ent_subj.text
                        # Only consider if we have a subject entity or a proper noun
                        if not subj_text:
                            continue
                        for obj in objects:
                            obj_text = obj.text
                            ent_obj = get_entity_for_token(obj, doc)
                            if ent_obj:
                                obj_text = ent_obj.text
                            if not obj_text:
                                continue
                            # Avoid self-loops
                            if subj_text == obj_text:
                                continue
                            # Determine relationship type using the verb's lemma.
                            rel_type = token.lemma_.lower()
                            # Adjust relation type for linking verbs if applicable.
                            if rel_type in ("be", "is", "was", "are", "were") and obj.dep_ in ("attr", "acomp", "oprd"):
                                rel_type = obj_text.lower().replace(" ", "_")
                            # Create a Single relationship and add to graph_list
                            graph_list.append(Single(node=subj_text, target_node=obj_text, relationship=rel_type))
    
    # Fallback: If no relationships were extracted via dependency parsing,
    # create relationships by linking consecutive entities in each sentence.
    if not graph_list:
        for sent in doc.sents:
            entities = list(sent.ents)
            if len(entities) >= 2:
                for i in range(len(entities) - 1):
                    node = entities[i].text.strip()
                    target_node = entities[i+1].text.strip()
                    relationship = "RELATED_TO"
                    graph_list.append(Single(node=node, target_node=target_node, relationship=relationship))
    
    return GraphComponents(graph=graph_list)

# ---------------------------
# For testing the module independently.
# ---------------------------
if __name__ == "__main__":
    text = "Alice and Bob founded Acme Corp in 2020. She later joined another startup."
    graph = spacy_llm_parser(text)
    print(graph.model_dump_json())
