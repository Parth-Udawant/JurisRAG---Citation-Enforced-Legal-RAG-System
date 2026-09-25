import os
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_ADMIN_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
INDEX_NAME = os.environ["AZURE_SEARCH_INDEX_NAME"]

OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

VECTOR_DIMENSIONS = int(
    os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536")
)

TOP = 5
VECTOR_K = 20

SEARCH_FIELDS = [
    "act",
    "chapter_title",
    "section_number",
    "section_title",
    "subsection",
    "content",
    "citation_text",
]

SELECT_FIELDS = [
    "chunk_id",
    "act",
    "chapter_number",
    "chapter_title",
    "section_number",
    "section_title",
    "subsection",
    "content_type",
    "content",
    "citation_id",
    "citation_text",
    "source_file",
]


search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_ADMIN_KEY),
)

embedding_client = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    azure_endpoint=OPENAI_ENDPOINT,
    api_version="2024-10-21",
)


def embed_query(query: str) -> list[float]:
    response = embedding_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=query,
        dimensions=VECTOR_DIMENSIONS,
    )

    vector = response.data[0].embedding

    if len(vector) != VECTOR_DIMENSIONS:
        raise RuntimeError(
            f"Expected query vector dimension {VECTOR_DIMENSIONS}, "
            f"got {len(vector)}"
        )

    return vector


def print_results(label: str, results) -> None:
    print(f"\n{'=' * 80}")
    print(label)
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        print(f"\n#{rank}")
        print(f"score      : {result.get('@search.score')}")
        print(f"chunk_id   : {result.get('chunk_id')}")
        print(f"act        : {result.get('act')}")
        print(f"section    : {result.get('section_number')}")
        print(f"citation   : {result.get('citation_text')}")
        print(f"content    : {result.get('content', '')[:500]}")


def run_query(query: str) -> None:
    print(f"\n\nQUERY: {query}")

    bm25_results = search_client.search(
        search_text=query,
        search_fields=SEARCH_FIELDS,
        select=SELECT_FIELDS,
        top=TOP,
    )
    print_results("BM25-ONLY", bm25_results)

    query_vector = embed_query(query)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=VECTOR_K,
        fields="content_vector",
    )

    vector_results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=SELECT_FIELDS,
        top=TOP,
    )
    print_results("VECTOR-ONLY", vector_results)

    hybrid_results = search_client.search(
        search_text=query,
        search_fields=SEARCH_FIELDS,
        vector_queries=[vector_query],
        select=SELECT_FIELDS,
        top=TOP,
    )
    print_results("HYBRID (BM25 + VECTOR + RRF)", hybrid_results)


def main() -> None:
    queries = [
        "What is the punishment for murder under the Bharatiya Nyaya Sanhita?",
        "When is a contract void under the Indian Contract Act?",
    ]

    for query in queries:
        run_query(query)


if __name__ == "__main__":
    main()
