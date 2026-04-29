# Default Project Design

This is the repository-level fallback `DESIGN.md` used when a project workspace does not define its own `DESIGN.md`.

Project-level rules always win over this file.

## Global defaults

- Prefer simple, testable modules over deep framework magic.
- Keep domain logic separate from transport or UI layers.
- Keep persistence logic behind a thin repository or service boundary.
- Favor explicit wiring and small functions over hidden side effects.
- Add tests for behavior changes and bug fixes.

## Stack defaults

### Python CRUD
- Stack: `fastapi` + `react` + `postgres` + `python`
- API uses routers + Pydantic schemas + dependency-injected DB sessions.
- SQLAlchemy models are persistence-only; never use them as request or response bodies.
- Frontend calls backend through typed API helpers.

### Flask Classic
- Stack: `flask` + `htmx` + `sqlite` + `python`
- Use app factory + blueprints + extension registry.
- htmx endpoints return HTML partials, not JSON unless explicitly API routes.
- Keep template fragments in `templates/partials/`.

### Django Full
- Stack: `django` + `none` + `postgres` + `python`
- One app per domain area; route via `include()`.
- Use migrations for schema changes, not ad-hoc table creation.
- Keep heavy business logic out of views.

### Node Modern
- Stack: `hono` + `svelte` + `sqlite` + `node`
- Keep server routes, DB schema, and UI components clearly separated.
- Use one DB client instance, shared by handlers.
- Use migrations for schema evolution.

### Next Full-stack
- Stack: `nextjs-api` + `react` + `postgres` + `node`
- Use App Router conventions (`app/`, `route.ts`, `page.tsx`).
- Reuse a singleton Prisma client.
- Default to Server Components; opt into client components only when needed.

### CLI Tool
- Stack: `none` + `none` + `json-file` + `python`
- Keep CLI parsing in `cli.py`; keep logic in `core.py`.
- Keep file IO in `store.py`.
- Ensure commands are scriptable and deterministic.

### Desktop App
- Stack: `avalonia` + `none` + `none` + `dotnet`
- Use MVVM: minimal code-behind, logic in ViewModels and Services.
- Favor constructor-injected services and async APIs.
- Keep UI state in ViewModels, not models.

## Workspace docs contract

When available, use these docs in priority order:

1. `FOXFORGE.md`
2. `CLAUDE.md`
3. `PATTERNS.md`
4. `DESIGN.md`
5. `CODE.md`
6. `ARCHITECTURE.md`

If project docs and this fallback conflict, project docs take precedence.
