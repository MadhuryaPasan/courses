#When you declare other function parameters that are not part of the path parameters, they are automatically interpreted as "query" parameters.

from fastapi import FastAPI

app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    """
    skip and limit are query parameters.
    url usage: http://127.0.0.1:8000/items/?skip=0&limit=10

    if only use `http://127.0.0.1:8000/items/` this will assign default parameters
    """
    return fake_items_db[skip : skip + limit]



# ======================================

# optional parameters

@app.get("/items2/{item_id}")
async def read_item(item_id: str, q: str | None = None):
    """
    usage
    http://127.0.0.1:8000/items2/r234
    http://127.0.0.1:8000/items2/r234?q=hello
    """
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}



# bool types

@app.get("/items/{item_id}")
async def read_item(item_id: str, q: str | None = None, short: bool = False):
    """
    usage:
    http://127.0.0.1:8000/items/foo?short=1
    http://127.0.0.1:8000/items/foo?short=True
    http://127.0.0.1:8000/items/foo?short=true
    http://127.0.0.1:8000/items/foo?short=on
    http://127.0.0.1:8000/items/foo?short=yes
    http://127.0.0.1:8000/items/foo?short=False
    """
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item



# ===================================

# multiple path parameters

@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int, item_id: str, q: str | None = None, short: bool = False
):

"""
usage:
http://127.0.0.1:8000/users/555/items/jlakjlet555
"""
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item