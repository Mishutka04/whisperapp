from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from src.swagger.router import router as router_swagger
app = FastAPI(
        root_path="/api",
        docs_url="/"
    )


origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router_swagger)

@app.get('/')
def hello():
    return 'Hello World00'


