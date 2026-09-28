from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--golden-set", required=True)
    parser.add_argument("--output", default="api_answer_eval_results.jsonl")
    args = parser.parse_args()

    data = json.loads(Path(args.golden_set).read_text(encoding="utf-8"))
    questions = data.get("questions", [])
    if not questions and isinstance(data.get("acts"), dict):
        for block in data["acts"].values():
            questions.extend(block.get("questions", []))

    output = Path(args.output)
    completed = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["question_id"]] = row

    with output.open("a", encoding="utf-8") as f:
        for q in questions:
            if q["id"] in completed:
                continue
            response = requests.post(
                f"{args.base_url.rstrip('/')}/api/v1/chat",
                json={"question": q["query"], "act": "all", "conversation_id": None},
                timeout=120,
            )
            response.raise_for_status()
            body = response.json()
            row = {
                "question_id": q["id"],
                "act": q["act"],
                "query": q["query"],
                "gold": q["gold"],
                "api_response": body,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(q["id"], "ok")


if __name__ == "__main__":
    main()
