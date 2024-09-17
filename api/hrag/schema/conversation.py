from pydantic import BaseModel


class ChatSchemaRequest(BaseModel):
    user_id: str
    message: str


class ChatSchemaResponse(BaseModel):
    message: str
