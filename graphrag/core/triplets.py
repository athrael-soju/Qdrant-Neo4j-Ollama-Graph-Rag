import re
import logging
import nltk
from typing import List, Tuple, Dict, Any
from graphrag.connectors.neo4j_connection import get_connection

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

# Initialize logger
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download("punkt", quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {str(e)}")

# Default model for triplet extraction
DEFAULT_TRIPLET_MODEL = "bew/t5_sentence_to_triplet_xl"


# Placeholder for the embedding function
def embed_query_on_gpu(text: str):
    """
    Replace this placeholder with your actual GPU-accelerated embedding code.
    For demonstration, it returns a dummy vector.
    """
    import numpy as np

    # Assuming the embedding dimension is 768; adjust as needed.
    return np.random.rand(768)


class TripletExtractor:
    """Extract subject-relation-object triplets from text and map them into a Neo4j knowledge graph."""

    def __init__(self, neo4j_conn=None, model_name=DEFAULT_TRIPLET_MODEL):
        """
        Initialize the triplet extractor.

        Args:
            neo4j_conn: A Neo4j connection instance.
            model_name: Hugging Face model name for triplet extraction.
        """
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            raise ImportError(
                "Transformers library is required for triplet extraction. "
                "Install with 'pip install transformers'"
            )

        # Check for PEFT
        try:
            from peft import PeftModel

            peft_available = True
        except ImportError:
            peft_available = False
            logger.warning(
                "PEFT library not available. If using adapter models, install with 'pip install peft'"
            )

        self.neo4j = neo4j_conn or get_connection()

        # Check if Neo4j supports vector search
        try:
            version_info = self.neo4j.run_query(
                "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition"
            )
            neo4j_version = None
            neo4j_edition = None

            for item in version_info:
                if item.get("name") == "Neo4j Kernel":
                    neo4j_version = item.get("versions", [""])[0]
                    neo4j_edition = item.get("edition", "")

            logger.info(
                f"Detected Neo4j version {neo4j_version}, edition {neo4j_edition}"
            )

            # Vector indexes require Neo4j 5.11+ Enterprise Edition
            self.supports_vector = False
            if neo4j_edition == "enterprise" and neo4j_version:
                major, minor = map(int, neo4j_version.split(".")[:2])
                if major > 5 or (major == 5 and minor >= 11):
                    self.supports_vector = True

            if not self.supports_vector:
                logger.warning(
                    "Vector search not supported in this Neo4j version/edition. Will use text-based fallback."
                )
        except Exception as e:
            logger.warning(
                f"Could not determine Neo4j version/vector support: {str(e)}"
            )
            self.supports_vector = False

        # Load the tokenizer and model with robust error handling
        logger.info(f"Loading triplet extraction model: {model_name}")
        try:
            # Special handling for the PEFT adapter model
            if model_name == "bew/t5_sentence_to_triplet_xl":
                base_model_name = "google/flan-t5-xl"
                logger.info(
                    f"Using base model {base_model_name} for PEFT adapter {model_name}"
                )

                # Load the tokenizer from the base model
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

                # Load the base model first
                logger.info(f"Loading base model {base_model_name}")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

                # If PEFT is available, load the adapter on top of the base model
                if peft_available:
                    from peft import PeftModel, PeftConfig

                    logger.info(f"Loading PEFT adapter {model_name}")

                    try:
                        # Try loading the PEFT adapter
                        self.model = PeftModel.from_pretrained(self.model, model_name)
                        logger.info("PEFT adapter loaded successfully")
                    except Exception as e:
                        logger.error(f"Error loading PEFT adapter: {str(e)}")
                        logger.warning("Continuing with base model only")
                else:
                    logger.warning("PEFT library not available. Using base model only.")
            else:
                # Standard model loading for non-adapter models
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            logger.info("Triplet extraction model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise

    def extract_triplets(self, sentence: str) -> List[Tuple[str, str, str]]:
        """
        Extract triplets from a sentence using the T5 model.

        Args:
            sentence: Input sentence.

        Returns:
            List of (subject, relation, object) triplets.
        """
        try:
            # Encode input sentence and generate model output
            inputs = self.tokenizer(sentence, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=64)
            # Do not skip special tokens so markers are preserved if present
            triplet_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            logger.debug(f"Raw triplet model output: {triplet_text}")

            triplets = []
            # If expected markers exist, use the original logic.
            if "<triplet>" in triplet_text:
                for segment in triplet_text.split("<triplet>"):
                    if segment.strip():
                        triple_content = (
                            segment.split("</triplet>")[0]
                            if "</triplet>" in segment
                            else segment
                        )
                        triple_content = triple_content.replace("<pad>", "")
                        if (
                            "<relation>" in triple_content
                            and "<object>" in triple_content
                        ):
                            subj = triple_content.split("<relation>")[0].strip()
                            rel = (
                                triple_content.split("<relation>")[1]
                                .split("<object>")[0]
                                .strip()
                            )
                            obj = triple_content.split("<object>")[1].strip()
                            if subj and rel and obj:
                                triplets.append((subj, rel, obj))
            else:
                # Heuristic: Remove <pad> and </s>, then split on two or more spaces.
                cleaned_text = (
                    triplet_text.replace("<pad>", "").replace("</s>", "").strip()
                )
                parts = re.split(r"\s{2,}", cleaned_text)
                if len(parts) == 3:
                    subj, rel, obj = (
                        parts[0].strip(),
                        parts[1].strip(),
                        parts[2].strip(),
                    )
                    triplets.append((subj, rel, obj))
                else:
                    logger.warning(
                        f"Unexpected triplet format for sentence: '{sentence}'. Model output: '{triplet_text}'"
                    )

            if not triplets:
                logger.warning(
                    f"No triplets extracted for sentence: '{sentence}'. Model output: '{triplet_text}'"
                )
            else:
                logger.debug(f"Extracted triplets: {triplets}")
            return triplets

        except Exception as e:
            logger.error(f"Error extracting triplets from sentence: {str(e)}")
            logger.error(f"Sentence: {sentence[:100]}...")
            return []

    def sanitize_relation(self, rel: str) -> str:
        """
        Sanitize the relation string to be a valid Neo4j relationship type.

        Args:
            rel: Relation string.

        Returns:
            Sanitized relation string.
        """
        rel_clean = re.sub(r"[^0-9a-zA-Z_ ]", "", rel)
        return rel_clean.replace(" ", "_").upper()

    def process_triplet(self, triplet: Tuple[str, str, str]) -> Any:
        """
        Process a single triplet: compute embeddings, search for similar nodes via vector queries,
        and merge the triplet into the Neo4j graph.

        Args:
            triplet: A tuple of (subject, predicate, object).

        Returns:
            Results from the final Neo4j query.
        """
        subject, predicate, object_ = triplet

        # Compute embeddings for each component
        subject_emb = embed_query_on_gpu(subject)
        predicate_emb = embed_query_on_gpu(predicate)
        object_emb = embed_query_on_gpu(object_)

        params = {
            "subject_emb": subject_emb.tolist(),
            "predicate_emb": predicate_emb.tolist(),
            "object_emb": object_emb.tolist(),
            "subject": subject,
            "predicate": predicate,
            "object": object_,
        }

        # Initialize variables to store similar nodes
        similarSubjects = []
        similarPredicates = []
        similarObjects = []

        if self.supports_vector:
            # Find similar subject nodes using vector search
            try:
                similarSubjects_query = """
                CALL {
                    CALL db.index.vector.queryNodes('vector_index_token', 10, $subject_emb)
                    YIELD node AS vectorNode, score as vectorScore
                    WITH vectorNode, vectorScore
                    WHERE vectorScore >= 0.96
                    RETURN collect(vectorNode) AS similarSubjects
                }
                WITH similarSubjects
                OPTIONAL MATCH (n:Token {name: toLower($subject)})
                WITH similarSubjects + CASE WHEN n IS NULL THEN [] ELSE [n] END AS allSubjects
                UNWIND allSubjects AS subject
                RETURN collect(subject) AS similarSubjects
                """
                similarSubjects = self.neo4j.run_query(similarSubjects_query, params)[
                    0
                ]["similarSubjects"]

                # Find similar predicate nodes
                similarPredicates_query = """
                CALL {
                    CALL db.index.vector.queryNodes('vector_index_token', 10, $predicate_emb)
                    YIELD node AS vectorNode, score as vectorScore
                    WITH vectorNode, vectorScore
                    WHERE vectorScore >= 0.96
                    RETURN collect(vectorNode) AS similarPredicates
                }
                WITH similarPredicates
                OPTIONAL MATCH (n:Token {name: toLower($predicate)})
                WITH similarPredicates + CASE WHEN n IS NULL THEN [] ELSE [n] END AS allPredicates
                UNWIND allPredicates AS predicate
                RETURN collect(predicate) AS similarPredicates
                """
                similarPredicates = self.neo4j.run_query(
                    similarPredicates_query, params
                )[0]["similarPredicates"]

                # Find similar object nodes
                similarObjects_query = """
                CALL {
                    CALL db.index.vector.queryNodes('vector_index_token', 10, $object_emb)
                    YIELD node AS vectorNode, score as vectorScore
                    WITH vectorNode, vectorScore
                    WHERE vectorScore >= 0.96
                    RETURN collect(vectorNode) AS similarObjects
                }
                WITH similarObjects
                OPTIONAL MATCH (n:Token {name: toLower($object)})
                WITH similarObjects + CASE WHEN n IS NULL THEN [] ELSE [n] END AS allObjects
                UNWIND allObjects AS object
                RETURN collect(object) AS similarObjects
                """
                similarObjects = self.neo4j.run_query(similarObjects_query, params)[0][
                    "similarObjects"
                ]
            except Exception as e:
                logger.error(
                    f"Vector search failed, falling back to exact match: {str(e)}"
                )
                self.supports_vector = False  # Disable for future calls

        # If vector search is not supported or failed, use exact match as fallback
        if not self.supports_vector:
            # Use exact match for subjects
            exact_subject_query = """
            OPTIONAL MATCH (n:Token {name: toLower($subject)})
            RETURN CASE WHEN n IS NULL THEN [] ELSE [n] END AS similarSubjects
            """
            similarSubjects = self.neo4j.run_query(exact_subject_query, params)[0][
                "similarSubjects"
            ]

            # Use exact match for predicates
            exact_predicate_query = """
            OPTIONAL MATCH (n:Token {name: toLower($predicate)})
            RETURN CASE WHEN n IS NULL THEN [] ELSE [n] END AS similarPredicates
            """
            similarPredicates = self.neo4j.run_query(exact_predicate_query, params)[0][
                "similarPredicates"
            ]

            # Use exact match for objects
            exact_object_query = """
            OPTIONAL MATCH (n:Token {name: toLower($object)})
            RETURN CASE WHEN n IS NULL THEN [] ELSE [n] END AS similarObjects
            """
            similarObjects = self.neo4j.run_query(exact_object_query, params)[0][
                "similarObjects"
            ]

        # If no similar nodes found, create empty collections to ensure we still create the triplet
        if not similarSubjects:
            similarSubjects = []
        if not similarPredicates:
            similarPredicates = []
        if not similarObjects:
            similarObjects = []

        # If any of the collections are empty, we need to ensure we create nodes
        create_query = """
        MERGE (subjectNode:Token {name: toLower($subject)})
        ON CREATE SET subjectNode.embeddings = $subject_emb, subjectNode.triplet_part = 'subject'
        ON MATCH SET subjectNode.triplet_part = 'subject'
        
        MERGE (objectNode:Token {name: toLower($object)})
        ON CREATE SET objectNode.embeddings = $object_emb, objectNode.triplet_part = 'object'
        ON MATCH SET objectNode.triplet_part = 'object'
        
        MERGE (subjectNode)-[r:predicate {name: toLower($predicate)}]->(objectNode)
        ON CREATE SET r.label = 'triplet', r.embeddings = $predicate_emb
        ON MATCH SET r.label = 'triplet'
        
        RETURN subjectNode.name AS subject, $predicate AS predicate, objectNode.name AS object
        """
        self.neo4j.run_query(create_query, params)

        # If we have similar nodes, also connect them
        if similarSubjects and similarPredicates and similarObjects:
            # Merge the triplet into the graph using the similar nodes found
            query = """
            UNWIND $similarSubjects AS subject
            UNWIND $similarPredicates AS predicate
            UNWIND $similarObjects AS object
            WITH subject.name AS subjectName, predicate.name AS predicateName, object.name AS objectName, subject, predicate, object
            MERGE (subjectNode:Token {name: toLower(subjectName)})
            ON CREATE SET subjectNode.embeddings = $subject_emb, subjectNode.triplet_part = 'subject'
            ON MATCH SET subjectNode.triplet_part = 'subject'
            MERGE (objectNode:Token {name: toLower(objectName)})
            ON CREATE SET objectNode.embeddings = $object_emb, objectNode.triplet_part = 'object'
            ON MATCH SET objectNode.triplet_part = 'object'
            MERGE (subjectNode)-[r:predicate {name: toLower(predicateName)}]->(objectNode)
                ON CREATE SET r.label = 'triplet', r.embeddings = $predicate_emb
                ON MATCH SET r.label = 'triplet'
            RETURN subjectName AS subject, predicateName AS predicate, objectName AS object
            """
            final_params = {
                "similarSubjects": similarSubjects,
                "similarPredicates": similarPredicates,
                "similarObjects": similarObjects,
                "subject_emb": subject_emb.tolist(),
                "predicate_emb": predicate_emb.tolist(),
                "object_emb": object_emb.tolist(),
            }
            results = self.neo4j.run_query(query, final_params)
        else:
            results = [{"subject": subject, "predicate": predicate, "object": object_}]

        logger.info(f"Processed triplet: {triplet}")
        return results

    def process_chunk(
        self, chunk_id: str, chunk_text: str
    ) -> List[Tuple[str, str, str]]:
        """
        Process a text chunk: split into sentences, extract triplets from each, and map each triplet into the graph.

        Args:
            chunk_id: Identifier for the text chunk.
            chunk_text: The text content of the chunk.

        Returns:
            List of extracted triplets.
        """
        logger.info(f"Processing chunk {chunk_id} for triplet extraction")
        sentences = nltk.sent_tokenize(chunk_text)
        all_triplets = []
        for sent in sentences:
            triplets = self.extract_triplets(sent)
            all_triplets.extend(triplets)
            for triplet in triplets:
                self.process_triplet(triplet)

        logger.info(
            f"Extracted and processed {len(all_triplets)} triplets from chunk {chunk_id}"
        )
        return all_triplets

    def process_chunks(
        self, chunks: List[Tuple[str, str]]
    ) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Process multiple text chunks.

        Args:
            chunks: List of tuples (chunk_id, chunk_text).

        Returns:
            Dictionary mapping each chunk_id to its list of triplets.
        """
        results = {}
        for chunk_id, chunk_text in chunks:
            triplets = self.process_chunk(chunk_id, chunk_text)
            results[chunk_id] = triplets
        return results


# Convenience functions


def extract_triplets(sentence: str) -> List[Tuple[str, str, str]]:
    extractor = TripletExtractor()
    return extractor.extract_triplets(sentence)


def sanitize_relation(rel: str) -> str:
    extractor = TripletExtractor()
    return extractor.sanitize_relation(rel)


def process_chunk(chunk_id: str, chunk_text: str) -> List[Tuple[str, str, str]]:
    extractor = TripletExtractor()
    return extractor.process_chunk(chunk_id, chunk_text)


def process_chunks(
    chunks: List[Tuple[str, str]],
) -> Dict[str, List[Tuple[str, str, str]]]:
    extractor = TripletExtractor()
    return extractor.process_chunks(chunks)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Demo with an example sentence
    example_sentence = "Hugging Face, Inc. is an American company that develops tools for building applications using machine learning."
    print("Extracting triplets from example sentence...")
    triplets = extract_triplets(example_sentence)
    print(f"Extracted {len(triplets)} triplets:")
    for subj, rel, obj in triplets:
        print(f"  ({subj}, {rel}, {obj})")
