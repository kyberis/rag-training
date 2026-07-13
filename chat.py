"""
Interactive CLI for talking to the DocPlanner RAG assistant.

Requires having run first:
    python -m src.ingest

Usage:
    python chat.py
"""
from src.rag import answer


def main():
    print("=== DocPlanner Support Assistant (RAG demo) ===")
    print("Escribi tu pregunta (o 'salir' para terminar).\n")
    while True:
        question = input("Vos: ").strip()
        if question.lower() in {"salir", "exit", "quit"}:
            break
        if not question:
            continue
        result = answer(question)
        print(f"\nAsistente: {result['answer']}")
        print(f"Fuentes: {', '.join(result['sources'])}\n")


if __name__ == "__main__":
    main()
