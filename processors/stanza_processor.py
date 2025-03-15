import stanza
from pydantic import BaseModel
from typing import List
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download Stanza model if not already installed
stanza.download('en')

# Initialize Stanza NLP pipeline
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse,ner,sentiment')


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


def extract_entities(doc):
    """
    Extracts named entities from the text using Stanza's NER.
    """
    entities = [(ent.text, ent.type) for ent in doc.ents]
    print("Extracted Entities:", entities)
    return entities


def extract_dependencies(doc):
    """
    Extracts meaningful subject-verb-object (SVO) triples from the text.
    """
    relations = []
    for sentence in doc.sentences:
        subject_map = {}
        object_map = {}
        verb_map = {}

        for word in sentence.words:
            # Store verbs
            if word.upos == "VERB":
                verb_map[word.id] = word.text
            
            # Store subjects (who is doing the action)
            if word.deprel in ["nsubj", "nsubjpass"]:
                subject_map[word.head] = word.text  # Head is the verb
            
            # Store objects (what is receiving the action)
            if word.deprel in ["dobj", "iobj", "nmod"]:
                object_map[word.head] = word.text  # Head is the verb

        # Match verbs with their subjects and objects
        for verb_id, verb in verb_map.items():
            subject = subject_map.get(verb_id)
            obj = object_map.get(verb_id)

            if subject and obj:
                relations.append((subject, verb, obj))

    print("Extracted Relationships:", relations)
    return relations




def stanza_llm_parser(text: str) -> GraphComponents:
    """
    Extracts graph components using Stanza's capabilities.
    """
    doc = nlp(text)
    graph_list = []

    # Extract named entities
    entities = extract_entities(doc)

    # Extract dependency relationships
    relations = extract_dependencies(doc)

    # Convert dependency relations to graph components
    for subj, verb, obj in relations:
        subj_entity = next((e[0] for e in entities if subj in e[0]), subj)
        obj_entity = next((e[0] for e in entities if obj in e[0]), obj)

        if subj_entity != obj_entity:
            graph_list.append(Single(node=subj_entity, target_node=obj_entity, relationship=verb.lower()))

    if not graph_list and len(entities) >= 2:
        for i in range(len(entities) - 1):
            graph_list.append(Single(node=entities[i][0], target_node=entities[i + 1][0], relationship="RELATED_TO"))

    return GraphComponents(graph=graph_list)


if __name__ == "__main__":
    test_text = "John works at Microsoft. The company develops Windows."
    result = stanza_llm_parser(test_text)
    print(result.model_dump_json(indent=2))
