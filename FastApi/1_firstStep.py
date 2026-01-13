# pip install "fastapi[standard]"

from fastapi import FastAPI

app = FastAPI() # this is the @app part

# get: fetching data: read only, (loading a webpage)
# post: sending data: creating new record (submitting a form, uploading a image)
# put: update existing data or replace data
# delete
@app.get("/")
async def root():
    return {"message": "Hello World"}


# fastapi dev 1_firstStep.py [this is a new way to run server]