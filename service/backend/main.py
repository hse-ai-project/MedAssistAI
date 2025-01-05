import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from api.api_route import router, logger


app = FastAPI(title="MedAssistAI")


class InfoResponse(BaseModel):
    status: str


@app.get("/participants", response_model=InfoResponse)
async def info_out() -> InfoResponse:
    "Метод, выводящий информацию об участниках проекта"

    info = """
        Васильев Даниил — @daniel_vasiliev, AristanD

        Мартынов Александр — @martynovall, alexmart811

        Черных Иван — @zzzippp, xfiniks

        Ляпин Данила — @danila_lyapin, Lak1n26

        Куратор:
        Малюшитский Кирилл — @malyushitsky, malyushitsky
    """
    logger.debug("Got request from client to /participants")
    return InfoResponse(status=info)


# Роутер с основными api запросами
app.include_router(router, prefix="/model")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
