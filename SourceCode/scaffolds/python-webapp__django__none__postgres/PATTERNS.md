# Stack patterns — Django + PostgreSQL

## Directory structure

```
myproject/           # Django project root (settings, urls, wsgi)
  settings.py
  urls.py            # Root URL conf — include each app's urls.py here
  wsgi.py
myapp/               # One Django app per feature domain
  models.py
  views.py
  urls.py            # App-level URL conf
  serializers.py     # DRF serializers (if using Django REST Framework)
  admin.py
  apps.py
  migrations/        # Auto-generated — never edit by hand
manage.py
requirements.txt
```

## One app per domain

Split features into Django apps: `users`, `milestones`, `goals`, `streaks`. Each app has its own `models.py`, `views.py`, `urls.py`.

```bash
python manage.py startapp milestones
```

Register in `INSTALLED_APPS` in `settings.py`.

## Models

```python
# milestones/models.py
from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Milestone(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="milestones")
    name = models.CharField(max_length=200)
    completed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
```

Always use `get_user_model()` rather than importing `User` directly.

## URL routing

```python
# myproject/urls.py
from django.urls import path, include

urlpatterns = [
    path("api/milestones/", include("milestones.urls")),
    path("api/users/", include("users.urls")),
]

# milestones/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("", views.MilestoneListView.as_view(), name="milestone-list"),
    path("<int:pk>/", views.MilestoneDetailView.as_view(), name="milestone-detail"),
]
```

## DRF serializers (if using Django REST Framework)

```python
# milestones/serializers.py
from rest_framework import serializers
from .models import Milestone

class MilestoneSerializer(serializers.ModelSerializer):
    class Meta:
        model = Milestone
        fields = ["id", "name", "completed", "created_at"]
        read_only_fields = ["id", "created_at"]
```

## Migrations — always run after model changes

```bash
python manage.py makemigrations
python manage.py migrate
```

Never edit migration files by hand. Never call `syncdb` or `create_all`.

## Settings pattern

```python
# settings.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key")
DEBUG = os.environ.get("DEBUG", "False") == "True"
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("DB_NAME", "myproject"),
        "USER": os.environ.get("DB_USER", "postgres"),
        "PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "HOST": os.environ.get("DB_HOST", "localhost"),
        "PORT": os.environ.get("DB_PORT", "5432"),
    }
}
```

## Naming conventions

- Apps: plural snake_case — `milestones`, `training_sessions`
- Models: singular PascalCase — `Milestone`, `TrainingSession`
- Views: `{Model}ListView`, `{Model}DetailView`, `{Model}CreateView`
- URL names: `{app}-{action}` — `milestone-list`, `milestone-detail`
- Serializers: `{Model}Serializer`

## Common mistakes to avoid

- Do NOT import `User` directly — use `get_user_model()`
- Do NOT skip migrations — always `makemigrations` after model changes
- Do NOT put business logic in views — use model methods or service functions
- Do NOT hardcode secrets — use environment variables
- Do NOT add app URLs directly in the root `urls.py` — always use `include()`
