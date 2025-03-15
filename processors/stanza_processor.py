import stanza
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download Stanza model if not already installed
stanza.download("en")

# Initialize Stanza NLP pipeline with additional processors
nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse,ner,sentiment")


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


def extract_entities(doc) -> List[Tuple[str, str]]:
    """
    Extracts named entities from the text using Stanza's NER.

    Args:
        doc: Stanza document

    Returns:
        List of tuples containing (entity_text, entity_type)
    """
    entities = []

    # Extract standard named entities
    for ent in doc.ents:
        entities.append((ent.text, ent.type))

    # Also extract nouns that might be entities but not caught by NER
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos in ["NOUN", "PROPN"] and not any(
                word.text in e[0] for e in entities
            ):
                # Check if it's not part of an existing entity
                entities.append((word.text, "NOUN"))

    logger.info(f"Extracted Entities: {entities}")
    return entities


def find_full_entity(word_id, sentence, entity_map):
    """
    Finds the full entity span for a given word by looking at its dependents.

    Args:
        word_id: ID of the head word
        sentence: Stanza sentence object
        entity_map: Map of word IDs to entity texts

    Returns:
        Full entity text
    """
    if word_id in entity_map:
        return entity_map[word_id]

    word = sentence.words[word_id - 1]  # Convert to 0-indexed

    # Get the word itself
    text = word.text

    # Find dependent words that modify this word (like adjectives, determiners)
    dependents = []
    for w in sentence.words:
        if w.head == word_id and w.deprel in ["amod", "compound", "det", "nummod"]:
            dependents.append((w.id, w.text))

    # Sort dependents by ID to maintain correct word order
    dependents.sort()

    # Construct the full entity text
    full_text = " ".join([dep[1] for dep in dependents] + [text])

    # Store in map for future reference
    entity_map[word_id] = full_text

    return full_text


def extract_dependencies(doc, entities) -> List[Tuple[str, str, str]]:
    """
    Extracts meaningful subject-verb-object (SVO) triples from the text,
    with enhanced entity resolution.

    Args:
        doc: Stanza document
        entities: List of extracted entities

    Returns:
        List of (subject, verb, object) triples
    """
    relations = []
    entity_texts = [e[0] for e in entities]

    for sentence in doc.sentences:
        subject_map = {}
        object_map = {}
        verb_map = {}
        entity_id_map = {}  # Map word IDs to full entity spans

        # First pass: identify potential subjects, verbs, and objects
        for word in sentence.words:
            # Store verbs and their lemmas for better relationship labeling
            if word.upos == "VERB":
                verb_map[word.id] = word.lemma

            # Store subjects with dependency to their verbs
            if word.deprel in ["nsubj", "nsubjpass", "csubj"]:
                # Get the full entity for the subject
                subj_text = find_full_entity(word.id, sentence, entity_id_map)
                subject_map[word.head] = subj_text

            # Store objects with dependency to their verbs
            if word.deprel in ["obj", "dobj", "iobj", "obl"]:
                # Get the full entity for the object
                obj_text = find_full_entity(word.id, sentence, entity_id_map)
                object_map[word.head] = obj_text

        # Second pass: collect prepositional objects for better relationship context
        for word in sentence.words:
            if word.deprel == "case" and word.head - 1 < len(sentence.words):
                head_word = sentence.words[word.head - 1]
                if head_word.head in verb_map:
                    # This is a prepositional phrase attached to a verb
                    # e.g., "works [at] Microsoft" - "at" is the case marker
                    prep_obj = find_full_entity(word.head, sentence, entity_id_map)
                    object_map[head_word.head] = prep_obj
                    # Add the preposition to the verb for better relationship description
                    verb_map[head_word.head] = f"{verb_map[head_word.head]} {word.text}"

        # Third pass: match verbs with their subjects and objects
        for verb_id, verb in verb_map.items():
            subject = subject_map.get(verb_id)
            obj = object_map.get(verb_id)

            if subject and obj:
                relations.append((subject, verb, obj))

        # Fourth pass: handle coreference-like relationships
        # E.g., "John works at Microsoft. The company develops Windows."
        if len(relations) >= 1:
            last_entities = set()
            for rel in relations:
                last_entities.add(rel[0])
                last_entities.add(rel[2])

            for i in range(1, len(sentence.words)):
                curr_word = sentence.words[i]
                if curr_word.upos in ["PRON", "DET"] and curr_word.text.lower() in [
                    "it",
                    "they",
                    "the",
                    "this",
                    "that",
                ]:
                    # Potential coreference - check if it's a subject or object
                    if curr_word.deprel in ["nsubj", "nsubjpass"]:
                        # This pronoun is likely referring to a previously mentioned entity
                        for entity in last_entities:
                            if entity in entity_texts:
                                subject_map[curr_word.head] = entity
                                break

    # Check for entity relationships not captured through verbs
    # For example, possessive relationships or appositive constructions
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.deprel in ["nmod:poss", "appos"]:
                # Get the head and dependent
                head_word = sentence.words[word.head - 1]
                head_text = find_full_entity(word.head, sentence, {})
                dep_text = find_full_entity(word.id, sentence, {})

                if word.deprel == "nmod:poss":  # Possessive relationship
                    relations.append((dep_text, "owns", head_text))
                elif word.deprel == "appos":  # Appositive relationship
                    relations.append((head_text, "is", dep_text))

    logger.info(f"Extracted Relationships: {relations}")
    return relations


def match_entity_with_text(text, entities):
    """
    Matches text with the most appropriate entity.

    Args:
        text: Text to match
        entities: List of extracted entities

    Returns:
        Best matching entity or the original text
    """
    # First try exact match
    for entity, entity_type in entities:
        if text.lower() == entity.lower():
            return entity

    # Then try contains match
    for entity, entity_type in entities:
        if text.lower() in entity.lower() or entity.lower() in text.lower():
            return entity

    return text


def stanza_llm_parser(text: str) -> GraphComponents:
    """
    Extracts graph components using enhanced Stanza capabilities.

    Args:
        text: Input text

    Returns:
        GraphComponents object containing the extracted graph
    """
    doc = nlp(text)
    graph_list = []

    # Extract named entities and other potential entities
    entities = extract_entities(doc)

    # Extract dependency relationships
    relations = extract_dependencies(doc, entities)

    # Convert dependency relations to graph components
    for subj, verb, obj in relations:
        # Match with extracted entities for consistency
        subj_entity = match_entity_with_text(subj, entities)
        obj_entity = match_entity_with_text(obj, entities)

        if subj_entity and obj_entity and subj_entity != obj_entity:
            graph_list.append(
                Single(
                    node=subj_entity, target_node=obj_entity, relationship=verb.lower()
                )
            )

    # Build a set of connected entities and a map of entity types
    connected_entities = set()
    entity_types = {}
    for rel in graph_list:
        connected_entities.add(rel.node)
        connected_entities.add(rel.target_node)

    for entity, entity_type in entities:
        entity_types[entity] = entity_type

    # Find disconnected entities
    disconnected = [e[0] for e in entities if e[0] not in connected_entities]

    # Strategy 1: Connect entities that appear close to each other in the text
    sentences = [sent.text for sent in doc.sentences]
    entity_positions = {}

    # Map entities to their positions in text
    for i, sentence in enumerate(sentences):
        for entity in [e[0] for e in entities]:
            if entity in sentence:
                if entity not in entity_positions:
                    entity_positions[entity] = []
                entity_positions[entity].append(i)

    # Connect entities that appear in the same or adjacent sentences
    for i, entity1 in enumerate([e[0] for e in entities]):
        for j, entity2 in enumerate([e[0] for e in entities][i + 1 :], i + 1):
            if entity1 != entity2:
                # Check if they appear in the same or adjacent sentences
                positions1 = entity_positions.get(entity1, [])
                positions2 = entity_positions.get(entity2, [])

                # Check for same or adjacent sentences
                for pos1 in positions1:
                    for pos2 in positions2:
                        if abs(pos1 - pos2) <= 1:  # Same or adjacent sentence
                            if entity1 in disconnected or entity2 in disconnected:
                                graph_list.append(
                                    Single(
                                        node=entity1,
                                        target_node=entity2,
                                        relationship="MENTIONED_NEAR",
                                    )
                                )
                                # Remove from disconnected if they were there
                                if entity1 in disconnected:
                                    disconnected.remove(entity1)
                                if entity2 in disconnected:
                                    disconnected.remove(entity2)
                                break
                    else:
                        continue
                    break

    # Strategy 2: Find central entities (entities with most connections)
    entity_connections = {}
    for rel in graph_list:
        if rel.node not in entity_connections:
            entity_connections[rel.node] = 0
        if rel.target_node not in entity_connections:
            entity_connections[rel.target_node] = 0
        entity_connections[rel.node] += 1
        entity_connections[rel.target_node] += 1

    # Sort entities by number of connections
    central_entities = sorted(
        entity_connections.items(), key=lambda x: x[1], reverse=True
    )

    # Connect remaining disconnected entities to central entities
    if central_entities and disconnected:
        main_entity = central_entities[0][0] if central_entities else None

        for entity in disconnected[:]:  # Use a copy for iteration
            if main_entity and entity != main_entity:
                graph_list.append(
                    Single(
                        node=entity, target_node=main_entity, relationship="RELATED_TO"
                    )
                )
                disconnected.remove(entity)

    # Strategy 3: Group entities by type and connect them
    # This helps with entities like locations, organizations, people
    entity_by_type = {}
    for entity, entity_type in entities:
        if entity_type not in entity_by_type:
            entity_by_type[entity_type] = []
        entity_by_type[entity_type].append(entity)

    # Connect entities of same type
    for entity_type, entities_of_type in entity_by_type.items():
        if len(entities_of_type) >= 2 and entity_type != "NOUN":
            for i in range(len(entities_of_type) - 1):
                # Only connect if one of them is disconnected
                if (
                    entities_of_type[i] in disconnected
                    or entities_of_type[i + 1] in disconnected
                ):
                    graph_list.append(
                        Single(
                            node=entities_of_type[i],
                            target_node=entities_of_type[i + 1],
                            relationship=f"SAME_{entity_type}",
                        )
                    )
                    # Remove from disconnected if they were there
                    if entities_of_type[i] in disconnected:
                        disconnected.remove(entities_of_type[i])
                    if entities_of_type[i + 1] in disconnected:
                        disconnected.remove(entities_of_type[i + 1])

    # Strategy 4: Use semantic similarity for related topics
    # This is a simple implementation - in practice, you might use word embeddings
    topic_groups = {
        "LOCATION": ["city", "country", "region", "area", "place", "capital"],
        "PERSON": ["person", "people", "individual", "human", "man", "woman"],
        "ORGANIZATION": ["company", "organization", "corporation", "firm", "business"],
        "EVENT": ["event", "meeting", "conference", "gathering", "ceremony"],
        "DATE": ["date", "time", "year", "month", "day"],
    }

    # Link entities that might be semantically related
    for entity1 in disconnected[:]:
        entity1_type = entity_types.get(entity1, "")
        for entity2 in [e[0] for e in entities]:
            if entity1 != entity2:
                entity2_type = entity_types.get(entity2, "")

                # Check if they belong to related topic groups
                for topic, keywords in topic_groups.items():
                    if any(keyword in entity1.lower() for keyword in keywords) and any(
                        keyword in entity2.lower() for keyword in keywords
                    ):
                        graph_list.append(
                            Single(
                                node=entity1,
                                target_node=entity2,
                                relationship=f"RELATED_{topic}",
                            )
                        )
                        if entity1 in disconnected:
                            disconnected.remove(entity1)
                        break

    # Strategy 5: Connect any remaining disconnected entities to their nearest entity in text
    for entity in disconnected:
        closest_entity = None
        closest_distance = float("inf")

        for other_entity in [e[0] for e in entities]:
            if entity != other_entity and other_entity not in disconnected:
                # Find minimum distance between entity positions
                if entity in entity_positions and other_entity in entity_positions:
                    for pos1 in entity_positions[entity]:
                        for pos2 in entity_positions[other_entity]:
                            distance = abs(pos1 - pos2)
                            if distance < closest_distance:
                                closest_distance = distance
                                closest_entity = other_entity

        if closest_entity:
            graph_list.append(
                Single(
                    node=entity,
                    target_node=closest_entity,
                    relationship="ASSOCIATED_WITH",
                )
            )

    return GraphComponents(graph=graph_list)


if __name__ == "__main__":
    test_text = "John works at Microsoft. The company develops Windows. Apple competes with Microsoft in the technology sector."
    result = stanza_llm_parser(test_text)
    print(result.model_dump_json(indent=2))
