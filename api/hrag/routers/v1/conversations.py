import logging
from http import HTTPStatus

from fastapi_versioning.versioning import version
from hrag.routers.router import conversations
from hrag.schema import ChatSchemaRequest, ChatSchemaResponse
from hrag.utils.conversation import (
    generate_chat_response,
    remove_user_chat_history,
)
from settings import app_settings

logger = logging.getLogger("gunicorn.error")
logger.setLevel(app_settings.log_level.upper())


@conversations.post(
    "/",
    response_model=ChatSchemaResponse,
    description="Generate response with Reception AI",
)
@version(1)
async def generate_response(tenant: str, chat: ChatSchemaRequest):
    message = await generate_chat_response(
        user_id=chat.user_id,
        message=chat.message,
        tenant=tenant,
    )
    return {"message": message}


@conversations.delete(
    "/{user_id}/",
    status_code=HTTPStatus.ACCEPTED,
    description="Remove old conversation with user to start a new session",
)
@version(1)
async def remove_old_conversation_with_user(tenant: str, user_id: str):
    await remove_user_chat_history(tenant=tenant, user_id=user_id)
