from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="graphrag",
    version="0.1.0",
    author="GraphRAG Team",
    author_email="example@example.com",
    description="Graph-based Retrieval Augmented Generation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/athrael-soju/graphrag",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "neo4j",
        "numpy",
        "nltk",
        "sentence-transformers",
        "transformers",
        "peft",
        "pymupdf",  # Optional: for PDF support
        "pyspark",  # Optional: for Spark NLP
        "spark-nlp",  # Optional: for Spark NLP 
        "qdrant-client",
    ],
    entry_points={
        "console_scripts": [
            "graphrag=graphrag.main:main",
        ],
    },
) 