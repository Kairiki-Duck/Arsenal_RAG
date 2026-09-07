import argparse
from pathlib import Path

from config import DOCUMENT_DIR, INDEX_PATH, METADATA_PATH


def build_parser():
    parser = argparse.ArgumentParser(description="Build the Arsenal RAG FAISS index")
    parser.add_argument("--document-dir", default=DOCUMENT_DIR, type=Path)
    parser.add_argument("--index-path", default=INDEX_PATH, type=Path)
    parser.add_argument("--metadata-path", default=METADATA_PATH, type=Path)
    parser.add_argument("--model-path", default=None, help="embedding model path or Hugging Face ID")
    parser.add_argument("--batch-size", default=32, type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")

    from src.chunker import split_markdown
    from src.embedding import EmbeddingModel
    from src.loader import load_documents
    from src.vector_store import VectorStore

    """
    Load documents
    """
    print("Loading documents...")
    documents = load_documents(args.document_dir)
    print(f"Loaded {len(documents)} documents.")

    """
    Chunk the documents
    """
    print("Chunking documents...")
    chunks=[]
    for document in documents:
        text=document["text"]
        source=document["source"]

        document_chunks=split_markdown(
            text
        )

        source_stem=Path(source).stem

        for chunk_index, chunk in enumerate(document_chunks):
            chunks.append({
                "chunk_id":f"{source_stem}_{chunk_index:06d}",
                "text":chunk["text"],
                "section":chunk["section"],
                "source":source,
                "chunk_index":chunk_index,
            })
    
    print(f"Created {len(chunks)} chunks.")

    """
    Load embedding model
    """
    print("Loading embedding model...")
    embedding_model = EmbeddingModel(model_path=args.model_path, batch_size=args.batch_size) \
        if args.model_path else EmbeddingModel(batch_size=args.batch_size)

    """
    Create embeddings
    """
    print("Creating embeddings...")
    texts=[
        chunk["text"] for chunk in chunks
    ]

    embeddings=embedding_model.encode(texts)

    print("Embedding shape:",embeddings.shape)

    """
    Create FAISS vector store
    """
    print("Creating FAISS vector store...")
    dimension=embeddings.shape[1]
    vector_store=VectorStore(dimension=dimension)
    vector_store.add(embeddings, chunks)

    """
    Save the vector store and metadata
    """
    print("Saving vector store and metadata...")
    vector_store.save(args.index_path, args.metadata_path)

    print("The knowledge index has been built and saved successfully.")
    print("FAISS index:", args.index_path)
    print("Metadata:", args.metadata_path)

if __name__=="__main__":
    main()