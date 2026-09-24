import os

from dotenv import load_dotenv
from openai import OpenAI


def main():
    load_dotenv()

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
    dimensions = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536"))

    required = {
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_API_KEY": api_key,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": deployment,
        "AZURE_OPENAI_EMBEDDING_MODEL": model,
    }

    missing = [name for name, value in required.items() if not value]

    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}"
        )

    base_url = f"{endpoint.rstrip('/')}/openai/v1/"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    test_text = (
        "Section 1 of the Bharatiya Nyaya Sanhita, 2023 "
        "deals with the short title, commencement and application."
    )

    print("Calling Azure embedding deployment...")
    print(f"Deployment: {deployment}")
    print(f"Model: {model}")
    print(f"Requested dimensions: {dimensions}")

    response = client.embeddings.create(
        model=deployment,
        input=test_text,
        dimensions=dimensions,
    )

    embedding = response.data[0].embedding

    print("\nEmbedding request successful.")
    print(f"Returned vector length: {len(embedding)}")
    print(f"Expected vector length: {dimensions}")

    if len(embedding) != dimensions:
        raise RuntimeError(
            f"Dimension mismatch: expected {dimensions}, "
            f"received {len(embedding)}"
        )

    print("Dimension check: PASS")
    print(f"First 5 values: {embedding[:5]}")


if __name__ == "__main__":
    main()