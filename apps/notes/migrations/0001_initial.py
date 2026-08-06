import uuid
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("workspaces", "0001_initial"),
        ("documents", "0001_initial"),
    ]
    operations = [
        migrations.CreateModel(
            name="Note",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("page_number", models.PositiveIntegerField(blank=True, null=True)),
                ("title", models.CharField(max_length=180)),
                ("content", models.TextField()),
                ("tags", models.CharField(blank=True, max_length=255)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("document", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="notes", to="documents.document")),
                ("owner", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="notes", to=settings.AUTH_USER_MODEL)),
                ("workspace", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="notes", to="workspaces.workspace")),
            ],
        )
    ]
