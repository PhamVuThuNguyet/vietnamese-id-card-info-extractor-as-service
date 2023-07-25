from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

from Sources.Controllers import main
handler = Mangum(app)