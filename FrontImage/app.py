import uvicorn
from Sources.Controllers.config import PORT

if __name__ == "__main__":
    uvicorn.run("Sources:app", host='0.0.0.0', port=int(PORT), reload=True)
