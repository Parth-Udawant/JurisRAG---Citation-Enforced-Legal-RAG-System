from __future__ import annotations

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

from ..core.config import Settings

SELECT_FIELDS = [
    "chunk_id", "act", "chapter_number", "chapter_title", "section_number",
    "section_title", "subsection", "content_type", "content", "citation_id",
    "citation_text", "source_file",
]


def _odata_escape(value: str) -> str:
    return value.replace("'", "''")


class SearchService:
    def __init__(self, settings: Settings, openai_client: AzureOpenAI):
        self.settings = settings
        self.openai_client = openai_client
        self.client = SearchClient(
            endpoint=settings.search_endpoint,
            index_name=settings.search_index_name,
            credential=AzureKeyCredential(settings.search_admin_key),
        )

    def embed_query(self, question: str) -> tuple[list[float], int]:
        response = self.openai_client.embeddings.create(
            model=self.settings.embedding_deployment,
            input=question,
            dimensions=self.settings.embedding_dimensions,
        )
        vector = response.data[0].embedding
        if len(vector) != self.settings.embedding_dimensions:
            raise RuntimeError("Query embedding dimension mismatch")
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        return vector, tokens

    def retrieve(self, question: str, act: str = "all") -> tuple[list[dict], int]:
        vector, embedding_tokens = self.embed_query(question)
        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=self.settings.candidate_k,
            fields="content_vector",
        )

        filter_expression = None
        if act == "bns":
            filter_expression = "act eq 'Bharatiya Nyaya Sanhita, 2023'"
        elif act == "ica":
            filter_expression = "act eq 'Indian Contract Act, 1872'"

        raw_results = self.client.search(
            search_text=question,
            vector_queries=[vector_query],
            filter=filter_expression,
            vector_filter_mode=self.settings.vector_filter_mode,
            select=SELECT_FIELDS,
            top=self.settings.candidate_k,
        )

        results = []
        for rank, result in enumerate(raw_results, start=1):
            item = dict(result)
            item["rank"] = rank
            item["search_score"] = result.get("@search.score")
            results.append(item)
            if len(results) >= self.settings.final_k:
                break
        return results, embedding_tokens
