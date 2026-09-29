import csv
import json
import os
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conversation_log.csv")


# --- Clientes HTTP hacia la API FastAPI local ---

def get_books() -> list:
    """Obtiene la lista completa de libros."""
    response = httpx.get(f"{API_BASE_URL}/books")
    response.raise_for_status()
    return response.json()


def get_books_alerts(threshold: int = 5) -> list:
    """Obtiene los libros con cantidad por debajo del umbral indicado."""
    response = httpx.get(f"{API_BASE_URL}/books/alerts", params={"threshold": threshold})
    response.raise_for_status()
    return response.json()


def create_book(titulo: str, autor: str, cantidad: int) -> dict:
    """Crea un nuevo libro en el catálogo."""
    response = httpx.post(
        f"{API_BASE_URL}/books",
        json={"titulo": titulo, "autor": autor, "cantidad": cantidad},
    )
    response.raise_for_status()
    return response.json()


def update_book_stock(book_id: int, delta: int) -> dict:
    """Actualiza el stock de un libro sumando/restando `delta` a la cantidad actual."""
    response = httpx.patch(f"{API_BASE_URL}/books/{book_id}", json={"delta": delta})
    response.raise_for_status()
    return response.json()


# --- Definición de tools en formato JSON schema compatible con Groq/OpenAI ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_books",
            "description": "Devuelve la lista completa de libros del catálogo, con su id, titulo, autor y cantidad.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_books_alerts",
            "description": "Devuelve los libros cuya cantidad en stock es menor a un umbral (por defecto 5).",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "integer",
                        "description": "Umbral de cantidad mínima. Libros por debajo de este valor se consideran en alerta.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_book",
            "description": "Crea un nuevo libro en el catálogo con título, autor y cantidad inicial.",
            "parameters": {
                "type": "object",
                "properties": {
                    "titulo": {"type": "string", "description": "Título del libro."},
                    "autor": {"type": "string", "description": "Autor del libro."},
                    "cantidad": {"type": "integer", "description": "Cantidad inicial en stock."},
                },
                "required": ["titulo", "autor", "cantidad"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_book_stock",
            "description": (
                "Actualiza el stock de un libro existente sumando un delta a la cantidad actual. "
                "Un delta positivo representa una reposición y un delta negativo representa una venta."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {"type": "integer", "description": "Id del libro a actualizar."},
                    "delta": {
                        "type": "integer",
                        "description": "Cambio a aplicar a la cantidad (positivo = reposición, negativo = venta).",
                    },
                },
                "required": ["book_id", "delta"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_books": get_books,
    "get_books_alerts": get_books_alerts,
    "create_book": create_book,
    "update_book_stock": update_book_stock,
}


# --- Registro de eventos en conversation_log.csv ---

def log_event(role: str, content: str = "", tool_name: str = "", tool_args=None, tool_result=None) -> None:
    is_new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["timestamp", "role", "content", "tool_name", "tool_args", "tool_result"])
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                role,
                content,
                tool_name,
                json.dumps(tool_args, ensure_ascii=False) if tool_args is not None else "",
                json.dumps(tool_result, ensure_ascii=False) if tool_result is not None else "",
            ]
        )


# --- Bucle del agente: observar -> pensar -> actuar -> actualizar ---

def run_agent() -> None:
    if not GROQ_API_KEY:
        raise RuntimeError("Falta GROQ_API_KEY. Definila en el archivo .env")

    client = Groq(api_key=GROQ_API_KEY)
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un asistente que gestiona el inventario de una librería. "
                "Usa las herramientas disponibles para consultar el catálogo, detectar alertas de stock bajo, "
                "crear libros y actualizar existencias."
            ),
        }
    ]

    print("Agente de gestión de librería (escribe 'salir' para terminar)")

    while True:
        # 1. Observar: leer la entrada del usuario por consola
        user_input = input("Tú: ").strip()
        if user_input.lower() in ("salir", "exit", "quit"):
            break

        log_event("user", user_input)
        messages.append({"role": "user", "content": user_input})

        while True:
            # 2. Pensar: enviar el historial y las tools al LLM
            completion = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=1,
                max_completion_tokens=2048,
                top_p=1,
            )
            message = completion.choices[0].message
            tool_calls = message.tool_calls

            if not tool_calls:
                # 5. Repetir hasta que el LLM dé una respuesta final (sin llamar a tool)
                final_answer = message.content or ""
                messages.append({"role": "assistant", "content": final_answer})
                log_event("agent", final_answer)
                print(f"Agente: {final_answer}")
                break

            # 3. Actuar: el LLM llamó a una o más tools, ejecutarlas contra la API
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                }
            )

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments or "{}")
                log_event("agent", tool_name=tool_name, tool_args=tool_args)

                func = TOOL_FUNCTIONS.get(tool_name)
                try:
                    result = func(**tool_args) if func else {"error": f"Tool desconocida: {tool_name}"}
                except Exception as exc:
                    result = {"error": str(exc)}

                log_event("tool", tool_name=tool_name, tool_args=tool_args, tool_result=result)

                # 4. Actualizar: inyectar el resultado como mensaje de rol "tool"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )


if __name__ == "__main__":
    run_agent()
