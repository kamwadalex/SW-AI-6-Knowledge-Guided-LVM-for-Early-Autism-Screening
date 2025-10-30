from fastapi import APIRouter
from pydantic import BaseModel


class EchoRequest(BaseModel):
	message: str


router = APIRouter()


@router.post("/echo")
def echo(payload: EchoRequest) -> dict:
	return {"echo": payload.message}


