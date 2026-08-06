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
            name="Conversation",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("title", models.CharField(default="New research conversation", max_length=180)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("owner", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="conversations", to=settings.AUTH_USER_MODEL)),
                ("workspace", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="conversations", to="workspaces.workspace")),
            ],
            options={"ordering": ["-updated_at"]},
        ),
        migrations.CreateModel(
            name="Message",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("role", models.CharField(choices=[("USER", "User"), ("ASSISTANT", "ScholarSync")], max_length=12)),
                ("content", models.TextField()),
                ("model_name", models.CharField(blank=True, max_length=100)),
                ("confidence", models.CharField(blank=True, max_length=20)),
                ("response_time_ms", models.PositiveIntegerField(default=0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("conversation", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="messages", to="conversations.conversation")),
            ],
            options={"ordering": ["created_at"]},
        ),
        migrations.CreateModel(
            name="Citation",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("page_number", models.PositiveIntegerField()),
                ("citation_number", models.PositiveIntegerField()),
                ("quoted_passage", models.TextField()),
                ("retrieval_score", models.FloatField(default=0)),
                ("verification_status", models.CharField(default="VALID", max_length=20)),
                ("chunk", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="citations", to="documents.documentchunk")),
                ("document", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="citations", to="documents.document")),
                ("message", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="citations", to="conversations.message")),
            ],
            options={"ordering": ["citation_number"]},
        ),
    ]
