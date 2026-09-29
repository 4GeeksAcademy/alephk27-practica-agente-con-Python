from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from app.data_init import init_books_csv, BOOKS_CSV

app = FastAPI(title="Gestión de Librerías con IA")


class Book(BaseModel):
    titulo: str
    autor: str
    cantidad: int


class BookStockUpdate(BaseModel):
    delta: int


@app.on_event("startup")
def startup_event():
    init_books_csv()


@app.get("/")
def root():
    return {"message": "API de gestión de librerías con IA"}


@app.get("/books", status_code=200)
def get_books():
    df = pd.read_csv(BOOKS_CSV)
    return df.to_dict(orient="records")


@app.get("/books/alerts", status_code=200)
def get_books_alerts(threshold: int = 5):
    df = pd.read_csv(BOOKS_CSV)
    alerts = df[df["cantidad"] < threshold]
    return alerts.to_dict(orient="records")


@app.get("/books/{book_id}", status_code=200)
def get_book(book_id: int):
    df = pd.read_csv(BOOKS_CSV)
    row = df[df["id"] == book_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Libro no encontrado")
    return row.to_dict(orient="records")[0]


@app.post("/books", status_code=201)
def create_book(book: Book):
    df = pd.read_csv(BOOKS_CSV)
    new_id = int(df["id"].max()) + 1 if not df.empty else 1
    new_row = {"id": new_id, **book.model_dump()}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(BOOKS_CSV, index=False)
    return new_row


@app.put("/books/{book_id}", status_code=200)
def update_book(book_id: int, book: Book):
    df = pd.read_csv(BOOKS_CSV)
    if book_id not in df["id"].values:
        raise HTTPException(status_code=404, detail="Libro no encontrado")
    df.loc[df["id"] == book_id, ["titulo", "autor", "cantidad"]] = [
        book.titulo,
        book.autor,
        book.cantidad,
    ]
    df.to_csv(BOOKS_CSV, index=False)
    return df[df["id"] == book_id].to_dict(orient="records")[0]


@app.patch("/books/{book_id}", status_code=200)
def update_stock(book_id: int, update: BookStockUpdate):
    df = pd.read_csv(BOOKS_CSV)
    if book_id not in df["id"].values:
        raise HTTPException(status_code=404, detail="Libro no encontrado")

    current = int(df.loc[df["id"] == book_id, "cantidad"].iloc[0])
    new_cantidad = current + update.delta
    if new_cantidad < 0:
        raise HTTPException(
            status_code=400,
            detail="La cantidad resultante no puede ser negativa",
        )

    df.loc[df["id"] == book_id, "cantidad"] = new_cantidad
    df.to_csv(BOOKS_CSV, index=False)
    return df[df["id"] == book_id].to_dict(orient="records")[0]


@app.delete("/books/{book_id}", status_code=200)
def delete_book(book_id: int):
    df = pd.read_csv(BOOKS_CSV)
    if book_id not in df["id"].values:
        raise HTTPException(status_code=404, detail="Libro no encontrado")
    df = df[df["id"] != book_id]
    df.to_csv(BOOKS_CSV, index=False)
    return {"message": f"Libro {book_id} eliminado"}
