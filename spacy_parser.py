import spacy
import re
from collections import defaultdict
from spacy.matcher import Matcher, DependencyMatcher

# Load spaCy model - you'll need to install it first with:
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def read_sample_data(file_path="sample_data.txt"):
    """Read the sample data from the specified file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_entities_and_relationships(text):
    """Extract entities and their relationships from the text using NLP with improved dependency parsing."""
    # Split the text into sentences for better processing
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    all_sentences = []
    for paragraph in paragraphs:
        doc = nlp(paragraph)
        all_sentences.extend([sent.text for sent in doc.sents])
    
    # Track all entities and their types
    entities = defaultdict(set)  # {entity_name: {entity_types}}
    relationships = []
    
    # First, get all entities from the text
    for paragraph in paragraphs:
        doc = nlp(paragraph)
        for ent in doc.ents:
            # Skip very short entities as they might be noise
            if len(ent.text) <= 2:
                continue
            # Store entity with its type
            entities[ent.text].add(ent.label_)
    
    # Now process each sentence for subject-verb-object patterns
    for sentence in all_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        doc = nlp(sentence)
        
        # Get entities in this sentence
        sentence_entities = {}
        for ent in doc.ents:
            if ent.text in entities:
                # Filter out very short entities or common words
                if len(ent.text) <= 2:
                    continue
                sentence_entities[ent.text] = next(iter(entities[ent.text]))
        
        # Extract subject-verb-object triplets
        extract_svo_triplets(doc, sentence_entities, relationships, sentence)
    
    # For entities that frequently co-occur, add association relationships
    add_cooccurrence_relationships_from_paragraphs(paragraphs, entities, relationships)
    
    # Convert defaultdict to regular dict
    entity_dict = {entity: list(types) for entity, types in entities.items()}
    
    return entity_dict, relationships

def extract_svo_triplets(doc, sentence_entities, relationships, context):
    """Extract subject-verb-object triplets from the sentence with improved handling of prepositional phrases."""
    # Create a dependency matcher for more precise matching of linguistic patterns
    dep_matcher = DependencyMatcher(nlp.vocab)
    
    # Pattern 1: Simple subject-verb-object
    pattern1 = [
        # Subject
        {
            "RIGHT_ID": "subject",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}
        },
        # Verb
        {
            "LEFT_ID": "subject",
            "REL_OP": ">",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"DEP": "ROOT", "POS": "VERB"}
        },
        # Direct object
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "object",
            "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "attr"]}}
        }
    ]
    
    # Pattern 2: Subject-verb-preposition-object (with proper connection between verb and preposition)
    pattern2 = [
        # Subject
        {
            "RIGHT_ID": "subject",
            "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}
        },
        # Verb
        {
            "LEFT_ID": "subject",
            "REL_OP": ">",
            "RIGHT_ID": "verb",
            "RIGHT_ATTRS": {"DEP": "ROOT", "POS": "VERB"}
        },
        # Preposition that's actually governed by the verb
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {"DEP": "prep"}
        },
        # Object of preposition
        {
            "LEFT_ID": "prep",
            "REL_OP": ">",
            "RIGHT_ID": "pobj",
            "RIGHT_ATTRS": {"DEP": "pobj"}
        }
    ]
    
    # Add patterns to matcher
    dep_matcher.add("SVO", [pattern1])
    dep_matcher.add("SVPO", [pattern2])
    
    # Get matches
    matches = dep_matcher(doc)
    
    for match_id, token_ids in matches:
        pattern_name = doc.vocab.strings[match_id]
        
        # Extract the subject, verb, and object based on pattern
        if pattern_name == "SVO":
            # Simple subject-verb-object pattern
            subj_token = doc[token_ids[0]]
            verb_token = doc[token_ids[1]]
            obj_token = doc[token_ids[2]]
            
            # Get full spans
            subj_span = get_full_span(subj_token)
            obj_span = get_full_span(obj_token)
            
            # Check if subject and object match entities
            subject_entity = find_entity_containing(subj_span.text, sentence_entities.keys())
            object_entity = find_entity_containing(obj_span.text, sentence_entities.keys())
            
            if subject_entity and object_entity and subject_entity != object_entity:
                verb = verb_token.lemma_.upper()
                
                # Skip certain generic verbs
                if not is_meaningful_verb(verb_token.lemma_):
                    continue
                
                # Create relationship
                add_relationship(
                    relationships, 
                    subject_entity, sentence_entities[subject_entity],
                    object_entity, sentence_entities[object_entity],
                    verb, context, 0.9
                )
                
        elif pattern_name == "SVPO":
            # Subject-verb-preposition-object pattern
            subj_token = doc[token_ids[0]]
            verb_token = doc[token_ids[1]]
            prep_token = doc[token_ids[2]]
            pobj_token = doc[token_ids[3]]
            
            # Get full spans
            subj_span = get_full_span(subj_token)
            pobj_span = get_full_span(pobj_token)
            
            # Check if subject and object match entities
            subject_entity = find_entity_containing(subj_span.text, sentence_entities.keys())
            object_entity = find_entity_containing(pobj_span.text, sentence_entities.keys())
            
            if subject_entity and object_entity and subject_entity != object_entity:
                # Check if the preposition is properly connected to the verb
                # (not just coincidentally appearing near it)
                if is_prep_modifying_verb(prep_token, verb_token):
                    verb_prep = f"{verb_token.lemma_}_{prep_token.text}".upper()
                    
                    # Skip certain generic verb-preposition combinations
                    if not is_meaningful_verb(verb_token.lemma_):
                        continue
                    
                    # Create relationship
                    add_relationship(
                        relationships, 
                        subject_entity, sentence_entities[subject_entity],
                        object_entity, sentence_entities[object_entity],
                        verb_prep, context, 0.9
                    )
    
    # Also use standard dependency parsing approach to catch relationships that might be missed
    extract_dependency_relationships(doc, sentence_entities, relationships, context)

def is_prep_modifying_verb(prep_token, verb_token):
    """Check if a preposition is actually modifying a given verb (not introducing a new clause)."""
    # Check if the preposition's head is the verb
    if prep_token.head == verb_token:
        return True
    
    # Check distance - in a legitimate verb+preposition, they should be close
    # This helps distinguish "Carol managed X with Dave" from 
    # "Carol managed X, with Dave being in charge"
    token_distance = abs(prep_token.i - verb_token.i)
    if token_distance > 5:  # If more than 5 tokens apart, likely not related
        return False
    
    # Check for commas or other clause separators between the verb and preposition
    if any(t.is_punct for t in verb_token.doc[verb_token.i:prep_token.i+1]):
        return False
    
    return True

def is_meaningful_verb(lemma):
    """Check if a verb is meaningful enough to form a relationship."""
    generic_verbs = {
        "be", "have", "do", "say", "get", "make", "go", "know", "will", 
        "think", "take", "see", "come", "want", "look", "use", "find", 
        "give", "tell", "work", "call", "try", "ask", "need", "seem", 
        "feel", "become", "leave", "put"
    }
    
    return lemma.lower() not in generic_verbs

def extract_dependency_relationships(doc, sentence_entities, relationships, context):
    """Extract relationships using dependency parsing with improved handling of prepositional phrases."""
    # Get all entity tokens in the document
    entity_tokens = {}
    for entity in sentence_entities:
        for token in doc:
            if entity.lower() in token.text.lower() or token.text.lower() in entity.lower():
                if entity not in entity_tokens:
                    entity_tokens[entity] = []
                entity_tokens[entity].append(token)
    
    # Find relationships between entities
    for entity1, tokens1 in entity_tokens.items():
        for entity2, tokens2 in entity_tokens.items():
            if entity1 == entity2:
                continue
                
            # Check each token pair for potential relationships
            for token1 in tokens1:
                for token2 in tokens2:
                    # Look for paths connecting the two tokens
                    relationship = find_relationship_in_dependency_path(token1, token2, doc)
                    
                    if relationship:
                        # Add the relationship if it's meaningful
                        add_relationship(
                            relationships,
                            entity1, sentence_entities[entity1],
                            entity2, sentence_entities[entity2],
                            relationship, context, 0.85
                        )

def find_relationship_in_dependency_path(token1, token2, doc):
    """Find a relationship verb along the dependency path between two tokens."""
    # Get the head token from each path toward root
    path1 = list(token1.ancestors)
    path2 = list(token2.ancestors)
    
    # Find common ancestor
    common_ancestors = set(path1).intersection(set(path2))
    if not common_ancestors:
        return None
    
    # Get the closest common ancestor
    common_ancestor = min(common_ancestors, key=lambda t: (path1.index(t) if t in path1 else float('inf')) + 
                                                       (path2.index(t) if t in path2 else float('inf')))
    
    # Check if the common ancestor is a verb
    if common_ancestor.pos_ == "VERB":
        # Get any prepositions that might be modifying this verb
        preps = [child for child in common_ancestor.children if child.dep_ == "prep"]
        
        # If there's a preposition connecting the verb to token2 or its ancestors
        for prep in preps:
            if prep in path2 or any(t in prep.subtree for t in path2):
                # Ensure this is a true verb+preposition combination, not separate clauses
                if is_prep_modifying_verb(prep, common_ancestor):
                    return f"{common_ancestor.lemma_}_{prep.text}".upper()
        
        # If no prepositions or no valid prep connection, return just the verb
        if is_meaningful_verb(common_ancestor.lemma_):
            return common_ancestor.lemma_.upper()
    
    return None

def add_relationship(relationships, source, source_type, target, target_type, rel_type, context, confidence):
    """Add a relationship to the list, avoiding duplicates."""
    # Check if this relationship already exists
    for rel in relationships:
        if (rel["source"] == source and rel["target"] == target and rel["type"] == rel_type) or \
           (rel["source"] == target and rel["target"] == source and rel["type"] == rel_type):
            # Update confidence if the new one is higher
            if confidence > rel.get("confidence", 0):
                rel["confidence"] = confidence
            return
    
    # Add new relationship if it doesn't exist
    relationships.append({
        "source": source,
        "source_type": source_type,
        "target": target,
        "target_type": target_type,
        "type": rel_type,
        "context": context,
        "confidence": confidence
    })

def get_full_span(token):
    """Get the full noun phrase span for a token, including all children."""
    min_idx = token.i
    max_idx = token.i
    
    for child in token.subtree:
        if child.dep_ in ("compound", "amod", "det", "nummod", "poss"):
            min_idx = min(min_idx, child.i)
            max_idx = max(max_idx, child.i)
    
    return token.doc[min_idx:max_idx+1]

def find_entity_containing(text, entities):
    """Find an entity that contains the given text."""
    text = text.lower().strip()
    
    # Direct match
    for entity in entities:
        if entity.lower() == text:
            return entity
    
    # Check if the text is contained in any entity
    for entity in entities:
        entity_words = entity.lower().split()
        text_words = text.split()
        
        # Check if all words in text are in entity
        if all(word in entity.lower() for word in text_words):
            return entity
        
        # Check if entity is in text
        if all(word in text for word in entity_words):
            return entity
    
    return None

def add_cooccurrence_relationships_from_paragraphs(paragraphs, entities, relationships):
    """Add co-occurrence relationships between entities that appear in the same paragraph."""
    # For each paragraph, find all entities
    for paragraph in paragraphs:
        doc = nlp(paragraph)
        
        # Get entities in this paragraph
        paragraph_entities = {}
        for ent in doc.ents:
            if ent.text in entities:
                paragraph_entities[ent.text] = next(iter(entities[ent.text]))
        
        # Create relationships between co-occurring entities
        entity_names = list(paragraph_entities.keys())
        for i, entity1 in enumerate(entity_names):
            for entity2 in entity_names[i+1:]:
                # Check if this entity pair already has a relationship
                has_relationship = False
                for rel in relationships:
                    if (rel["source"] == entity1 and rel["target"] == entity2) or \
                       (rel["source"] == entity2 and rel["target"] == entity1):
                        has_relationship = True
                        break
                
                # If no relationship exists, add an ASSOCIATED_WITH relationship
                if not has_relationship and entity1 != entity2:
                    relationships.append({
                        "source": entity1,
                        "source_type": paragraph_entities[entity1],
                        "target": entity2,
                        "target_type": paragraph_entities[entity2],
                        "type": "ASSOCIATED_WITH",
                        "context": paragraph[:150] + "..." if len(paragraph) > 150 else paragraph,
                        "confidence": 0.5  # Lower confidence for co-occurrence
                    })

def format_for_neo4j(entities, relationships):
    """Format the extracted entities and relationships for Neo4j."""
    # Create Cypher queries for nodes (entities)
    node_queries = []
    for entity, types in entities.items():
        # Use the first entity type as the primary label
        primary_type = types[0] if types else "Entity"
        
        # Escape any single quotes in the entity name
        safe_entity = entity.replace("'", "\\'")
        
        # Create node with all its types as labels
        labels = ":".join([primary_type] + [t for t in types if t != primary_type])
        node_queries.append(f"CREATE (:{labels} {{name: '{safe_entity}', type: '{primary_type}'}});")
    
    # Create Cypher queries for relationships
    relationship_queries = []
    for rel in relationships:
        # Escape any single quotes
        source = rel["source"].replace("'", "\\'")
        target = rel["target"].replace("'", "\\'")
        context = rel["context"].replace("'", "\\'")
        confidence = rel.get("confidence", 0.7)  # Default confidence if not specified
        
        # Create the relationship with confidence
        relationship_queries.append(
            f"MATCH (a {{name: '{source}'}}), (b {{name: '{target}'}})\n"
            f"CREATE (a)-[:{rel['type']} {{context: '{context}', confidence: {confidence}}}]->(b);"
        )
    
    return node_queries, relationship_queries

def main():
    # Read the sample data
    text = read_sample_data()
    
    # Extract entities and relationships
    entities, relationships = extract_entities_and_relationships(text)
    
    print(f"Found {len(entities)} entities and {len(relationships)} relationships.")
    
    # Print entities
    print("\nEntities:")
    for entity, types in entities.items():
        print(f"- {entity} ({', '.join(types)})")
    
    # Print relationships
    print("\nRelationships:")
    for rel in relationships:
        confidence = rel.get("confidence", 0.7)
        print(f"- {rel['source']} ({rel['source_type']}) --[{rel['type']} {confidence:.1f}]--> {rel['target']} ({rel['target_type']})")
    
    # Format for Neo4j
    node_queries, relationship_queries = format_for_neo4j(entities, relationships)
    
    # Print Neo4j queries
    print("\nNeo4j Node Creation Queries:")
    for query in node_queries[:10]:  # Limit to 10 to avoid overwhelming output
        print(query)
    
    if len(node_queries) > 10:
        print(f"... and {len(node_queries) - 10} more node queries")
    
    print("\nNeo4j Relationship Creation Queries:")
    for query in relationship_queries[:10]:  # Limit to 10
        print(query)
    
    if len(relationship_queries) > 10:
        print(f"... and {len(relationship_queries) - 10} more relationship queries")

if __name__ == "__main__":
    main() 