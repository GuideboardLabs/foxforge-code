# Stack patterns — FastAPI + React + PostgreSQL

## Directory structure

```
backend/
  app/
    models/          # SQLAlchemy ORM models ONLY — never used as request bodies
    schemas/         # Pydantic models for request/response validation
    crud/            # Database operations — pure functions, accept db: Session
    routes/          # One file per resource (users.py, items.py, etc.)
    core/
      deps.py        # get_db generator, auth dependencies
      config.py      # Settings via pydantic-settings
    main.py          # FastAPI app init, router registration
    database.py      # engine, SessionLocal, Base
  alembic/           # Migrations — never run Base.metadata.create_all in production
  requirements.txt
frontend/
  src/
    components/      # Reusable UI components
    pages/           # Route-level page components
    api/             # Axios/fetch wrappers for backend endpoints
    store/           # State management (Redux/Zustand/Context)
    App.jsx
  public/
  package.json
```

## The SQLAlchemy / Pydantic split — this is the #1 gotcha

SQLAlchemy models (`models/`) define database tables. They are NEVER used as FastAPI request bodies.

Pydantic schemas (`schemas/`) define what the API accepts and returns. Always create at least:
- `UserCreate` (input — no id, no created_at)
- `UserResponse` (output — includes id, timestamps)
- `UserUpdate` (partial fields, all Optional)

```python
# schemas/user.py
from pydantic import BaseModel
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    email: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    model_config = {"from_attributes": True}   # allows ORM → Pydantic conversion
```

## get_db dependency — always lives in core/deps.py

```python
# core/deps.py
from app.database import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Every route that needs the DB uses `db: Session = Depends(get_db)`. Never import SessionLocal directly in routes.

## Router pattern — one file per resource

```python
# routes/users.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.deps import get_db
from app.schemas.user import UserCreate, UserResponse
from app import crud

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/", response_model=UserResponse, status_code=201)
def create_user(body: UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(db, body)
```

## main.py — router registration

```python
from fastapi import FastAPI
from app.routes import users, items, auth   # list every router here

app = FastAPI()
app.include_router(users.router)
app.include_router(items.router)
app.include_router(auth.router)
```

Never add route handlers directly to `app` — always use routers.

## database.py

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass
```

`Base.metadata.create_all(bind=engine)` is only used in tests or local dev scripts. In production, use Alembic migrations.

## CRUD layer — crud/ contains only db operations

```python
# crud/user.py
from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.user import UserCreate

def create_user(db: Session, data: UserCreate) -> User:
    obj = User(**data.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj

def get_user(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()
```

## Naming conventions

- Models: singular PascalCase — `User`, `Milestone`, `TrainingSession`
- Schemas: `{Model}Create`, `{Model}Response`, `{Model}Update`
- Routes: plural snake_case endpoints — `/users/`, `/training-sessions/`
- CRUD functions: `create_{model}`, `get_{model}`, `list_{models}`, `update_{model}`, `delete_{model}`
- Route files: plural snake_case — `users.py`, `training_sessions.py`

## Common mistakes to avoid

- Do NOT pass a SQLAlchemy model as a `response_model` — use a Pydantic schema
- Do NOT call `db.commit()` in routes — do it in CRUD functions
- Do NOT import `SessionLocal` in routes — use `Depends(get_db)`
- Do NOT put logic in `main.py` — it only registers routers and middleware
- Do NOT forget `model_config = {"from_attributes": True}` on response schemas
