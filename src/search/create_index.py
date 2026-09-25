import os
from pathlib import Path

from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_ADMIN_KEY = os.getenv("AZURE_SEARCH_ADMIN_KEY")
INDEX_NAME = os.getenv(
    "AZURE_SEARCH_INDEX_NAME",
    "legal-rag-index",
)

VECTOR_DIMENSIONS = 1536

if not SEARCH_ENDPOINT:
    raise RuntimeError(
        "Missing AZURE_SEARCH_ENDPOINT in .env"
    )

if not SEARCH_ADMIN_KEY:
    raise RuntimeError(
        "Missing AZURE_SEARCH_ADMIN_KEY in .env"
    )

index_client = SearchIndexClient(
    endpoint=SEARCH_ENDPOINT,
    credential=AzureKeyCredential(SEARCH_ADMIN_KEY),
)

hnsw_algorithm = HnswAlgorithmConfiguration(
    name="legal-hnsw",
    parameters=HnswParameters(
        m=4,
        ef_construction=400,
        ef_search=500,
        metric="cosine",
    ),
)

vector_profile = VectorSearchProfile(
    name="legal-vector-profile",
    algorithm_configuration_name="legal-hnsw",
)

vector_search = VectorSearch(
    algorithms=[
        hnsw_algorithm,
    ],
    profiles=[
        vector_profile,
    ],
)

fields = [

    SimpleField(
        name="chunk_id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
        retrievable=True,
    ),

    SearchableField(
        name="act",
        type=SearchFieldDataType.String,
        filterable=True,
        facetable=True,
        retrievable=True,
    ),

    SimpleField(
        name="chapter_number",
        type=SearchFieldDataType.String,
        filterable=True,
        retrievable=True,
    ),

    SearchableField(
        name="chapter_title",
        type=SearchFieldDataType.String,
        filterable=True,
        retrievable=True,
    ),

    SearchableField(
        name="section_number",
        type=SearchFieldDataType.String,
        filterable=True,
        retrievable=True,
    ),

    SearchableField(
        name="section_title",
        type=SearchFieldDataType.String,
        filterable=True,
        retrievable=True,
    ),

    SearchableField(
        name="subsection",
        type=SearchFieldDataType.String,
        retrievable=True,
    ),

    SimpleField(
        name="content_type",
        type=SearchFieldDataType.String,
        filterable=True,
        facetable=True,
        retrievable=True,
    ),

    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        retrievable=True,
    ),

    SimpleField(
        name="citation_id",
        type=SearchFieldDataType.String,
        filterable=True,
        retrievable=True,
    ),

    SearchableField(
        name="citation_text",
        type=SearchFieldDataType.String,
        filterable=True,
        retrievable=True,
    ),

    SimpleField(
        name="source_file",
        type=SearchFieldDataType.String,
        filterable=True,
        retrievable=True,
    ),

    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(
            SearchFieldDataType.Single
        ),
        searchable=True,
        retrievable=False,
        vector_search_dimensions=VECTOR_DIMENSIONS,
        vector_search_profile_name="legal-vector-profile",
    ),
]

index = SearchIndex(
    name=INDEX_NAME,
    fields=fields,
    vector_search=vector_search,
)

print("=" * 70)
print("Creating Azure AI Search index")
print("=" * 70)

print(f"Endpoint    : {SEARCH_ENDPOINT}")
print(f"Index       : {INDEX_NAME}")
print(f"Dimensions  : {VECTOR_DIMENSIONS}")
print("Algorithm   : HNSW")
print("Metric      : cosine")

result = index_client.create_or_update_index(index)

print()
print(f"Index created successfully: {result.name}")
print()
print("Fields:")
for field in result.fields:
    print(f"  - {field.name}")

print()
print("=" * 70)
print("INDEX CREATION COMPLETE")
print("=" * 70)