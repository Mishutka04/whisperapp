from fastapi import Depends, FastAPI
from src.swagger.router import router as router_swagger
app = FastAPI()




app.include_router(router_swagger)

@app.get('/')
def hello():
    return 'Hello World00'


