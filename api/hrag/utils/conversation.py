import logging

from hrag.models import Tenant, User
from hrag.utils.graph import HybridRagGraph
from settings import app_settings

logger = logging.getLogger("gunicorn.error")
logger.setLevel(app_settings.log_level.upper())


async def generate_chat_response(user_id: str, message: str, tenant: str):
    logger.debug(f"message: {message}")
    logger.debug(f"tenant: {tenant}")
    logger.debug(f"user_id {user_id}")
    tenant_obj = Tenant.get_tenant(tenant)

    user_obj = User.get_or_create_user(
        username=user_id,
        tenant_id=tenant_obj.id,
        create_if_not_exist=True,
        include_inactive=False,
    )

    graph = HybridRagGraph(
        tenant_obj,
        user_obj.chat_history,
        user_obj.chat_summary,
    )

    response = graph.generate_response(message)

    user_obj.update_chat_history(
        user_message=message,
        ai_message=response,
        new_summary=graph.memory.buffer,
    )
    return response


async def remove_user_chat_history(tenant: str, user_id: str):
    tenant_obj = Tenant.get_tenant(tenant)
    User.delete_user(username=user_id, tenant_id=tenant_obj.id)
