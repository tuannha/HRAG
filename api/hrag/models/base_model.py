import uuid

import sqlalchemy as db
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func


class BaseModel(object):
    id = db.Column(
        UUID(as_uuid=True),
        index=True,
        primary_key=True,
        unique=True,
    )
    status = db.Column(db.Boolean, default=True, nullable=False, index=True)
    created_dt = db.Column(db.DateTime(timezone=True), server_default=func.now())
    updated_dt = db.Column(
        db.DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    def __init__(self, status: bool = True):
        super().__init__()

        self.id = uuid.uuid4()
        self.status = status
