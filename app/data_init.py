import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
BOOKS_CSV = os.path.join(DATA_DIR, "books.csv")


def init_books_csv():
    """Create data/books.csv with dummy data if it doesn't exist yet."""
    if os.path.exists(BOOKS_CSV):
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.DataFrame(
        [
            {"id": 1, "titulo": "Cien años de soledad", "autor": "Gabriel García Márquez", "cantidad": 5},
            {"id": 2, "titulo": "Don Quijote de la Mancha", "autor": "Miguel de Cervantes", "cantidad": 3},
        ],
        columns=["id", "titulo", "autor", "cantidad"],
    )
    df.to_csv(BOOKS_CSV, index=False)
