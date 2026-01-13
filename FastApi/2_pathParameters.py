from fastapi import FastAPI
from enum import Enum
app = FastAPI()

# a path can be use only one time
# make sure to keep this order if not the `me` will get as a {item_id}
@app.get("/users/me")
async def read_user_me():
    return{"message":"the current user"}

@app.get("/item/{item_id}")
async def read_item(item_id:int):
    return{"item_id": item_id}

# =======================================================
# use this to predefine values. only these can be used.
# validations

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}


# ==================================================
# get path 
# without this `:path` the api will try to find a empty api url
# eg. `home/johndoe/myfile.txt` if you want this path without `:path` the fast api will look for a url that starting from `/files/home`

@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}