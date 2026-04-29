# Stack patterns — Flask + htmx + SQLite

## Directory structure

```
app/
  __init__.py        # create_app() factory, register blueprints, init extensions
  models.py          # SQLAlchemy models
  blueprints/
    auth.py          # Blueprint for auth routes
    main.py          # Blueprint for main/index routes
    api.py           # Blueprint for AJAX/htmx endpoints
  templates/
    base.html        # Base layout with htmx CDN script tag
    partials/        # htmx partial HTML fragments (returned by AJAX routes)
    auth/
    main/
  static/
    css/
    js/
  extensions.py      # db = SQLAlchemy(), login_manager = LoginManager(), etc.
config.py            # Config classes (DevelopmentConfig, ProductionConfig)
run.py               # Entry point: app = create_app(); app.run()
```

## Application factory pattern

```python
# app/__init__.py
from flask import Flask
from app.extensions import db, login_manager

def create_app(config_name="development"):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    db.init_app(app)
    login_manager.init_app(app)
    from app.blueprints.main import main as main_bp
    from app.blueprints.auth import auth as auth_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    return app
```

Never create the Flask app at module level outside of the factory.

## Extensions — always in extensions.py

```python
# app/extensions.py
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()
```

Import `db` from `app.extensions` everywhere — never from `app` directly (circular import).

## htmx patterns

htmx routes return HTML fragments, not JSON. Use `render_template("partials/item_row.html", item=item)`.

Key htmx attributes in templates:
- `hx-post="/items/"` — submit to route
- `hx-get="/items/{{id}}"` — fetch and swap
- `hx-target="#container"` — where to swap the response
- `hx-swap="outerHTML"` — replace the element itself

```html
<!-- templates/partials/item_row.html -->
<tr id="item-{{item.id}}">
  <td>{{item.name}}</td>
  <td>
    <button hx-delete="/items/{{item.id}}"
            hx-target="#item-{{item.id}}"
            hx-swap="outerHTML">Delete</button>
  </td>
</tr>
```

Full-page routes return full templates. htmx endpoints return partials.

## SQLite session scope

```python
# app/extensions.py
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()
# Flask-SQLAlchemy handles session teardown automatically — no manual db.close()
```

`db.session` is request-scoped. Use `db.session.add()`, `db.session.commit()`. Flask-SQLAlchemy removes the session at end of request.

## Blueprint pattern

```python
# app/blueprints/main.py
from flask import Blueprint, render_template
from app.models import Item

main = Blueprint("main", __name__)

@main.route("/")
def index():
    items = Item.query.all()
    return render_template("main/index.html", items=items)
```

## Naming conventions

- Blueprints: lowercase singular — `auth`, `main`, `api`
- Templates: `blueprintname/action.html` — `auth/login.html`, `main/index.html`
- Partials: `partials/resource_fragment.html`
- Models: singular PascalCase — `User`, `Item`
- Routes: snake_case with blueprint prefix — `main.index`, `auth.login`

## Common mistakes to avoid

- Do NOT use `db` before `db.init_app(app)` is called
- Do NOT import from `app` (the package) in models — import `db` from `extensions`
- Do NOT return JSON from htmx routes — return rendered HTML fragments
- Do NOT call `db.create_all()` in production — use Flask-Migrate
- Do NOT put route handlers in `__init__.py` — use blueprints
