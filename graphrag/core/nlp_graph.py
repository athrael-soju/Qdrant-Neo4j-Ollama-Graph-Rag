"""
NLP graph creation utilities for GraphRAG
"""

import logging
import nltk
from typing import List, Tuple, Optional
from itertools import chain
from graphrag.connectors.neo4j_connection import get_connection

# Initialize logger
logger = logging.getLogger(__name__)

# Download NLTK resources if not already downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {str(e)}")

# Get stopwords
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    # Fallback basic stopwords if NLTK stopwords not available
    STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
                'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
                'to', 'from', 'in', 'on', 'at', 'by', 'with'}

class NLPGraphBuilder:
    """Builds NLP-enhanced knowledge graph from text chunks"""
    
    def __init__(self, neo4j_conn=None, remove_stopwords=True):
        """Initialize NLPGraphBuilder
        
        Args:
            neo4j_conn: Neo4j connection instance
            remove_stopwords: Whether to remove stopwords from unigrams
        """
        self.neo4j = neo4j_conn or get_connection()
        self.remove_stopwords = remove_stopwords
        logger.info("Initialized NLPGraphBuilder")
        
    def extract_ngrams(self, text: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract unigrams, bigrams, and trigrams from text
        
        Args:
            text: Input text
            
        Returns:
            Tuple[List[str], List[str], List[str]]: Tuple of (unigrams, bigrams, trigrams)
        """
        # Tokenize and normalize text
        tokens = [w.lower() for w in nltk.word_tokenize(text) if w.isalnum()]
        
        # Filter stopwords if required
        if self.remove_stopwords:
            unigrams = [t for t in tokens if t not in STOPWORDS]
        else:
            unigrams = tokens
            
        # Generate bigrams and trigrams
        bigrams = [' '.join(b) for b in nltk.bigrams(tokens)]
        trigrams = [' '.join(t) for t in nltk.trigrams(tokens)]
        
        logger.debug(f"Extracted {len(unigrams)} unigrams, {len(bigrams)} bigrams, {len(trigrams)} trigrams")
        return unigrams, bigrams, trigrams
        
    def store_terms_for_chunk(self, chunk_id: str, unigrams: List[str], 
                          bigrams: List[str], trigrams: List[str]) -> None:
        """Store terms (n-grams) and connect them to their chunk in Neo4j
        
        Args:
            chunk_id: Chunk identifier
            unigrams: List of unigrams (single tokens)
            bigrams: List of bigrams (two-word phrases)
            trigrams: List of trigrams (three-word phrases)
        """
        logger.info(f"Storing terms for chunk {chunk_id}")
        
        # Combine all terms with a type indicator
        terms = [(t, "unigram") for t in unigrams] + \
                [(t, "bigram") for t in bigrams] + \
                [(t, "trigram") for t in trigrams]
                
        batch_size = 100  # Process in batches to avoid large transactions
        for i in range(0, len(terms), batch_size):
            batch = terms[i:i+batch_size]
            
            # Use a single parameterized query for the batch
            params = {
                "chunk_id": chunk_id,
                "terms": [{"text": term, "type": term_type} for term, term_type in batch]
            }
            
            try:
                # Use a query that doesn't involve vector operations
                self.neo4j.run_query(
                    """
                    MATCH (c:Chunk {id: $chunk_id})
                    UNWIND $terms AS term
                    MERGE (t:Term {text: term.text, type: term.type})
                    MERGE (c)-[:HAS_TERM]->(t)
                    """,
                    params
                )
                
                logger.debug(f"Stored batch of {len(batch)} terms for chunk {chunk_id}")
            except Exception as e:
                logger.error(f"Error storing terms batch for chunk {chunk_id}: {str(e)}")
                raise
                
        logger.info(f"Successfully stored all terms for chunk {chunk_id}")
        
    def process_chunk(self, chunk_id: str, chunk_text: str) -> Tuple[List[str], List[str], List[str]]:
        """Process a text chunk, extract n-grams, and store in Neo4j
        
        Args:
            chunk_id: Chunk identifier
            chunk_text: Chunk text content
            
        Returns:
            Tuple[List[str], List[str], List[str]]: Tuple of (unigrams, bigrams, trigrams)
        """
        logger.info(f"Processing chunk {chunk_id}")
        
        # Extract n-grams
        unigrams, bigrams, trigrams = self.extract_ngrams(chunk_text)
        
        # Store in Neo4j
        self.store_terms_for_chunk(chunk_id, unigrams, bigrams, trigrams)
        
        return unigrams, bigrams, trigrams

try:
    # Check if pyspark and spark-nlp are available
    import pyspark
    import sparknlp
    
    SPARK_AVAILABLE = True
    logger.info("PySpark and Spark NLP are available. Spark-based processing enabled.")
    
    class SparkNLPGraphBuilder(NLPGraphBuilder):
        """NLP Graph Builder using Spark NLP for more scalable processing"""
        
        def __init__(self, neo4j_conn=None, remove_stopwords=True):
            """Initialize SparkNLPGraphBuilder
            
            Args:
                neo4j_conn: Neo4j connection instance
                remove_stopwords: Whether to remove stopwords from unigrams
            """
            super().__init__(neo4j_conn, remove_stopwords)
            
            # Initialize Spark session with Spark NLP
            self.spark = sparknlp.start()
            
            # Import required Spark NLP components
            from sparknlp.base import DocumentAssembler, Finisher
            from sparknlp.annotator import Tokenizer, Normalizer, NGramGenerator
            from pyspark.ml import Pipeline
            
            # Define Spark NLP pipeline
            document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
            tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
            normalizer = Normalizer().setInputCols(["token"]).setOutputCol("normalized").setLowercase(True)
            
            # Generate unigrams (normalized tokens), bigrams, trigrams
            bigram_generator = NGramGenerator().setInputCols(["normalized"]).setOutputCol("bigrams").setN(2)
            trigram_generator = NGramGenerator().setInputCols(["normalized"]).setOutputCol("trigrams").setN(3)
            
            # Finisher to output results as Python lists
            finisher_unigram = Finisher().setInputCols(["normalized"]).setOutputCols(["tokens_out"])
            finisher_bigram = Finisher().setInputCols(["bigrams"]).setOutputCols(["bigrams_out"])
            finisher_trigram = Finisher().setInputCols(["trigrams"]).setOutputCols(["trigrams_out"])
            
            # Create pipeline
            self.pipeline = Pipeline(stages=[
                document_assembler, tokenizer, normalizer,
                bigram_generator, trigram_generator,
                finisher_unigram, finisher_bigram, finisher_trigram
            ])
            
            # Fit the pipeline (this creates a model that can transform data)
            self.model = self.pipeline.fit(self.spark.createDataFrame([("",)], ["text"]))
            
            logger.info("Spark NLP pipeline initialized")
            
        def process_chunks(self, chunks: List[Tuple[str, str]]) -> List[Tuple[str, List[str], List[str], List[str]]]:
            """Process multiple chunks with Spark NLP
            
            Args:
                chunks: List of (chunk_id, chunk_text) tuples
                
            Returns:
                List[Tuple[str, List[str], List[str], List[str]]]: List of (chunk_id, unigrams, bigrams, trigrams) tuples
            """
            logger.info(f"Processing {len(chunks)} chunks with Spark NLP")
            
            # Create Spark DataFrame from chunks
            chunk_df = self.spark.createDataFrame([(cid, txt) for cid, txt in chunks], ["id", "text"])
            
            # Apply the pipeline to transform the data
            result_df = self.model.transform(chunk_df)
            
            # Collect results and process
            processed_chunks = []
            for row in result_df.select("id", "tokens_out", "bigrams_out", "trigrams_out").collect():
                chunk_id = row["id"]
                unigrams = row["tokens_out"]
                bigrams = row["bigrams_out"]
                trigrams = row["trigrams_out"]
                
                # Remove stopwords if required
                if self.remove_stopwords:
                    unigrams = [t for t in unigrams if t not in STOPWORDS]
                
                processed_chunks.append((chunk_id, unigrams, bigrams, trigrams))
                
                # Store in Neo4j
                self.store_terms_for_chunk(chunk_id, unigrams, bigrams, trigrams)
                
            logger.info(f"Successfully processed {len(chunks)} chunks with Spark NLP")
            return processed_chunks
            
except ImportError:
    SPARK_AVAILABLE = False
    logger.info("PySpark and/or Spark NLP not available. Using NLTK for processing.")
    SparkNLPGraphBuilder = None

# Convenience functions

def extract_ngrams(text: str) -> Tuple[List[str], List[str], List[str]]:
    """Extract unigrams, bigrams, and trigrams from text
    
    Args:
        text: Input text
        
    Returns:
        Tuple[List[str], List[str], List[str]]: Tuple of (unigrams, bigrams, trigrams)
    """
    builder = NLPGraphBuilder()
    return builder.extract_ngrams(text)
    
def store_terms_for_chunk(chunk_id: str, unigrams: List[str], 
                      bigrams: List[str], trigrams: List[str]) -> None:
    """Store terms (n-grams) and connect them to their chunk in Neo4j
    
    Args:
        chunk_id: Chunk identifier
        unigrams: List of unigrams (single tokens)
        bigrams: List of bigrams (two-word phrases)
        trigrams: List of trigrams (three-word phrases)
    """
    builder = NLPGraphBuilder()
    builder.store_terms_for_chunk(chunk_id, unigrams, bigrams, trigrams)
    
def process_chunk(chunk_id: str, chunk_text: str) -> Tuple[List[str], List[str], List[str]]:
    """Process a text chunk, extract n-grams, and store in Neo4j
    
    Args:
        chunk_id: Chunk identifier
        chunk_text: Chunk text content
        
    Returns:
        Tuple[List[str], List[str], List[str]]: Tuple of (unigrams, bigrams, trigrams)
    """
    builder = NLPGraphBuilder()
    return builder.process_chunk(chunk_id, chunk_text)
    
def process_chunks_with_spark(chunks: List[Tuple[str, str]]) -> List[Tuple[str, List[str], List[str], List[str]]]:
    """Process multiple chunks with Spark NLP
    
    Args:
        chunks: List of (chunk_id, chunk_text) tuples
        
    Returns:
        List[Tuple[str, List[str], List[str], List[str]]]: List of (chunk_id, unigrams, bigrams, trigrams) tuples
    """
    if not SPARK_AVAILABLE:
        logger.warning("Spark NLP is not available. Falling back to sequential processing.")
        results = []
        for chunk_id, chunk_text in chunks:
            unigrams, bigrams, trigrams = process_chunk(chunk_id, chunk_text)
            results.append((chunk_id, unigrams, bigrams, trigrams))
        return results
        
    builder = SparkNLPGraphBuilder()
    return builder.process_chunks(chunks)
    
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Demo with example text
    example_text = """
    Hugging Face, Inc. is an American company that develops tools for building applications using machine learning.
    It was founded in 2016 and its headquarters is in New York City. The company is known for its libraries in natural
    language processing (NLP) and its platform that allows users to share machine learning models and datasets.
    """
    
    print("Extracting n-grams from example text...")
    unigrams, bigrams, trigrams = extract_ngrams(example_text)
    print(f"Extracted {len(unigrams)} unigrams, {len(bigrams)} bigrams, {len(trigrams)} trigrams")
    print("Sample unigrams:", unigrams[:5])
    print("Sample bigrams:", bigrams[:5])
    print("Sample trigrams:", trigrams[:5]) 