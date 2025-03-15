import time
import os
from dotenv import load_dotenv
from graph_rag import (
    initialize_clients,
    create_collection,
    extract_graph_components,
    extract_graph_components_parallel,
    ingest_to_neo4j,
    ingest_to_qdrant,
    retriever_search,
    fetch_related_graph,
    format_graph_context,
    graphRAG_run,
    clear_data,
    VECTOR_DIMENSION  # This value is set based on the provider in processor_factory
)

# Load environment variables from .env
load_dotenv()

if __name__ == "__main__":
    print("GraphRAG Interactive Console")
    print("Loading environment variables and initializing clients...")

    # Initialize clients
    neo4j_driver, qdrant_client, collection_name = initialize_clients()
    print("Clients initialized")

    # Load optimization parameters from environment, with defaults
    parallel_processing = os.getenv("PARALLEL_PROCESSING", "true").lower() == "true"
    max_workers = int(os.getenv("MAX_WORKERS", "8"))
    batch_size = int(os.getenv("BATCH_SIZE", "100"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "5000"))
    use_streaming = os.getenv("USE_STREAMING", "true").lower() == "true"
    # Remove any surrounding quotes and lower the model provider name
    model_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "openai").strip("'").lower()

    # Vector dimension is imported (set by the processor_factory)
    vector_dimension = VECTOR_DIMENSION

    # Ensure collection exists
    create_collection(qdrant_client, collection_name, vector_dimension)

    # Import processor components based on the selected provider
    from processors.processor_factory import get_processor
    processor = get_processor()
    # NOTE: If EXTRACTOR is enabled, only the extraction (llm_parser) and its related models are overridden.
    llm_parser = processor["llm_parser"]
    embeddings = processor["embeddings"]
    embeddings_batch = processor["embeddings_batch"]
    graphrag_query = processor["graphrag_query"]
    GraphComponents = processor["GraphComponents"]
    Single = processor["Single"]
    LLM_MODEL = processor["LLM_MODEL"]
    EMBEDDING_MODEL = processor["EMBEDDING_MODEL"]
    EXTRACTOR = processor["EXTRACTOR"]
    # Interactive console loop
    while True:
        print("\n" + "=" * 50)
        print("GraphRAG Console - Choose an option:")
        print("1. Ingest data")
        print("2. Clear all data")
        print("3. Ask a question")
        print("4. Configure settings")
        print("5. Exit")
        print("=" * 50)
        print(f"Current LLM provider: {model_provider.upper()}")
        print(f"Current Inference Model: {LLM_MODEL}")
        print(f"Current Embedding Model: {EMBEDDING_MODEL}")
        print(f"Current Extractor: {EXTRACTOR}")
        print("=" * 50)

        choice = input("Enter your choice (1-5): ")

        if choice == "1":
            print("\nIngesting Data")
            print("-" * 30)
            try:
                start_time = time.time()
                with open('sample_data.txt', 'r', encoding='utf-8') as file:
                    raw_data = file.read()

                if not raw_data.strip():
                    print("No data found in file. Returning to menu.")
                    continue

                print("Extracting graph components...")
                if parallel_processing:
                    nodes, relationships = extract_graph_components_parallel(raw_data, chunk_size=chunk_size, max_workers=max_workers)
                else:
                    nodes, relationships = extract_graph_components(raw_data)

                print(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships")

                print("Ingesting to Neo4j...")
                node_id_mapping = ingest_to_neo4j(neo4j_driver, nodes, relationships, batch_size=batch_size)
                print("Neo4j ingestion complete")

                print("Ingesting to Qdrant...")
                ingest_to_qdrant(qdrant_client, collection_name, raw_data, node_id_mapping)
                print("Qdrant ingestion complete")

                end_time = time.time()
                processing_time = end_time - start_time
                print(f"\nTotal processing time: {processing_time:.2f} seconds")

            except FileNotFoundError:
                print("Error: sample_data.txt file not found. Please ensure the file exists in the current directory.")
            except Exception as e:
                print(f"Error reading file: {str(e)}")

        elif choice == "2":
            print("\nClearing All Data")
            confirm = input("Are you sure you want to clear all data? (y/n): ")
            if confirm.lower() == 'y':
                print("Clearing data...")
                clear_data(neo4j_driver, qdrant_client, collection_name)
                print("All data cleared successfully")
            else:
                print("Data clearing cancelled")

        elif choice == "3":
            print("\nAsk a Question")
            query = input("Enter your question: ")

            if not query.strip():
                print("No question entered. Returning to menu.")
                continue

            start_time = time.time()
            print("Starting retriever search...")
            retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)

            if not hasattr(retriever_result, 'items') or not retriever_result.items:
                print("No results found. Try ingesting some data first.")
                continue

            print("Extracting entity IDs...")
            entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in retriever_result.items]

            print("Fetching related graph...")
            subgraph = fetch_related_graph(neo4j_driver, entity_ids)

            print("Formatting graph context...")
            graph_context = format_graph_context(subgraph)

            print("Running GraphRAG...")
            start_answer_time = time.time()

            # Refresh the provider setting if needed
            model_provider = os.getenv("DEFAULT_MODEL_PROVIDER", "openai").strip("'").lower()

            # Use streaming response
            stream_response = graphRAG_run(graph_context, query, stream=use_streaming)

            print("\nAnswer: ", end="", flush=True)
            full_answer = ""

            try:
                if use_streaming:
                    if model_provider == "openai":
                        for chunk in stream_response:
                            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                print(content, end="", flush=True)
                                full_answer += content
                    else:
                        for content in stream_response:
                            print(content, end="", flush=True)
                            full_answer += content
                else:
                    full_answer = stream_response
                    print(full_answer)
            except Exception as e:
                error_msg = f"Error processing response: {str(e)}"
                print(error_msg)
                full_answer = error_msg

            print()
            end_time = time.time()
            query_time = end_time - start_time
            answer_time = end_time - start_answer_time
            print(f"Query processing time: {query_time:.2f} seconds (Answer generation: {answer_time:.2f} seconds)")

        elif choice == "4":
            print("\nConfigure Optimization Settings")
            print("-" * 30)
            print("Current settings:")
            print(f"1. Parallel processing: {parallel_processing}")
            print(f"2. Number of workers: {max_workers}")
            print(f"3. Batch size: {batch_size}")
            print(f"4. Chunk size: {chunk_size}")
            print(f"5. Response streaming: {use_streaming}")
            print("6. Return to main menu")

            setting_choice = input("\nSelect setting to change (1-6): ")

            if setting_choice == "1":
                parallel_choice = input("Enable parallel processing? (y/n): ")
                parallel_processing = parallel_choice.lower() == 'y'
                print(f"Parallel processing {'enabled' if parallel_processing else 'disabled'}")
            elif setting_choice == "2":
                try:
                    new_workers = int(input(f"Enter number of workers (current: {max_workers}): "))
                    if new_workers > 0:
                        max_workers = new_workers
                        print(f"Workers set to {max_workers}")
                    else:
                        print("Number of workers must be positive")
                except ValueError:
                    print("Invalid input, must be a number")
            elif setting_choice == "3":
                try:
                    new_batch_size = int(input(f"Enter batch size (current: {batch_size}): "))
                    if new_batch_size > 0:
                        batch_size = new_batch_size
                        print(f"Batch size set to {batch_size}")
                    else:
                        print("Batch size must be positive")
                except ValueError:
                    print("Invalid input, must be a number")
            elif setting_choice == "4":
                try:
                    new_chunk_size = int(input(f"Enter chunk size in characters (current: {chunk_size}): "))
                    if new_chunk_size > 0:
                        chunk_size = new_chunk_size
                        print(f"Chunk size set to {chunk_size}")
                    else:
                        print("Chunk size must be positive")
                except ValueError:
                    print("Invalid input, must be a number")
            elif setting_choice == "5":
                streaming_choice = input("Enable response streaming? (y/n): ")
                use_streaming = streaming_choice.lower() == 'y'
                print(f"Response streaming {'enabled' if use_streaming else 'disabled'}")
            elif setting_choice == "6":
                pass
            else:
                print("Invalid choice. Please enter a number between 1 and 6.")

        elif choice == "5":
            print("Exiting GraphRAG Console. Goodbye!")
            neo4j_driver.close()
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")
