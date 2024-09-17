import logging
import uuid

import sqlalchemy as db
from fastapi.exceptions import HTTPException
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from .base import Base
from .base_model import BaseModel

log = logging.getLogger("gunicorn.error")


class User(BaseModel, Base):
    __tablename__ = "user"
    _primary_key_names = ["id"]

    username = db.Column(
        db.Unicode(35),
        nullable=False,
        index=True,
    )
    tenant_id = db.Column(
        UUID(as_uuid=True), db.ForeignKey("tenant.id"), nullable=False, index=True
    )
    chat_history = db.Column(
        JSONB(),
        nullable=True,
    )
    chat_summary = db.Column(
        db.UnicodeText(),
        nullable=True,
    )

    tenant = relationship(
        "Tenant", primaryjoin="Tenant.id == User.tenant_id", backref="users"
    )

    __table_args__ = (db.UniqueConstraint("username", "tenant_id"),)

    def __init__(
        self,
        username: str,
        tenant_id: uuid.UUID,
        status: bool = True,
    ):
        super().__init__(status=status)
        self.tenant_id = tenant_id
        self.username = username

    @classmethod
    def get_or_create_user(
        cls,
        username: str,
        tenant_id: uuid.UUID,
        create_if_not_exist: bool = True,
        include_inactive: bool = False,
    ):
        from fastapi_sqlalchemy import db as db_session

        query = db_session.session.query(User).filter(
            User.tenant_id == tenant_id,
            User.username == username,
        )
        if not include_inactive:
            query = query.filter(User.status == True)

        user = query.first()

        if not user and create_if_not_exist:
            log.debug(f"Unknown user {username}, creating a default one ...")
            user = cls.create_user(username=username, tenant_id=tenant_id)

        if not user:
            log.debug("Unknown or inactive tenant")
            raise HTTPException(status_code=404, detail="No active user found.")

        return user

    @classmethod
    def create_user(cls, username: str, tenant_id: uuid.UUID):
        from fastapi_sqlalchemy import db as db_session

        user = cls(username=username, tenant_id=tenant_id)
        db_session.session.add(user)
        db_session.session.flush()

        return user

    def update_chat_history(self, user_message: str, ai_message: str, new_summary: str):
        from fastapi_sqlalchemy import db as db_session

        # must convert to list before appending new history, otherwise sqlalchemy will not update chat_history column
        chat_history = list(self.chat_history) if self.chat_history else []

        chat_history += [
            {"role": "human", "content": user_message},
            {"role": "ai", "content": ai_message},
        ]
        self.chat_history = chat_history
        self.chat_summary = new_summary
        log.debug(f"chat history: {chat_history}")
        log.debug(f"chat_summary: {new_summary}")

        db_session.session.add(self)
        db_session.session.flush()

    @classmethod
    def delete_user(cls, username: str, tenant_id: uuid.UUID):
        from fastapi_sqlalchemy import db as db_session

        user = cls.get_or_create_user(
            username=username,
            tenant_id=tenant_id,
            create_if_not_exist=False,
            include_inactive=True,
        )
        db_session.session.delete(user)
        db_session.session.flush()
