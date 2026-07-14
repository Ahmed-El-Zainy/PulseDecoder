from fastapi import FastAPI, Body

app = FastAPI()
books = [
    {"category": "Science", "title": "chimstry", "author": "John Doe"},
    {"category": "Sport", "title": "football", "author": "Jane Smith"},
    {"category": "History", "title": "world war II", "author": "Demtry Alan"},
]

# class Body:
#     category: str
#     title: str
#     author: str

@app.get("/books")
async def read_all_books():
    return books

@app.get("/books/category/{category}")
async def read_category_by_query(category: str):
    books_to_return = [ ]
    for book in books:
        if book.get("category").casefold() == category.casefold():
            books_to_return.append(book)
    return books_to_return



@app.post("/books/create_books/")
async def add_book(new_book=Body()):
    books.append(new_book)
    return {"message": "Book added successfully"}



@app.get("/books/{book_author}")
async def read_author_category_by_query(book_author:str, category: str):
    books_to_return = [ ]
    for book in books :
        if book.get("author").casefold() == book_author.casefold() and \
        book.get("category").casefold() == category.casefold():
            books_to_return.append(book)
    return books_to_return

@app.put("/books/update_book/{book_author}/{category}")
async def update_book(book_author: str, category: str, new_book=Body()):
    for book in books:
        if book.get("author").casefold() == book_author.casefold() and \
        book.get("category").casefold() == category.casefold():
            book.update(new_book)
            return {"message": "Book updated successfully"}
    return {"message": "Book not found"}

@app.delete("/books/delete_book/{book_author}/{category}")
async def delete_book(book_author: str, category: str):
    for book in books:
        if book.get("author").casefold() == book_author.casefold() and \
        book.get("category").casefold() == category.casefold():
            books.remove(book)
            return {"message": "Book deleted successfully"}
    return {"message": "Book not found"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
