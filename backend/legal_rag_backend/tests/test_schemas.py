from legal_rag_backend.models.schemas import ChatRequest


def test_chat_request_defaults():
    request = ChatRequest(question="What is murder?")
    assert request.act == "all"
    assert request.conversation_id is None
