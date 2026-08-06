import uuid
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("workspaces", "0001_initial"),
    ]
    operations = [
        migrations.CreateModel(
            name="Document",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("original_filename", models.CharField(max_length=255)),
                ("display_title", models.CharField(max_length=255)),
                ("storage_path", models.CharField(max_length=500)),
                ("file_hash", models.CharField(db_index=True, max_length=64)),
                ("file_size", models.PositiveBigIntegerField(default=0)),
                ("page_count", models.PositiveIntegerField(default=0)),
                ("mime_type", models.CharField(default="application/pdf", max_length=100)),
                ("processing_status", models.CharField(choices=[("UPLOADED", "Uploaded"), ("EXTRACTING", "Extracting"), ("CHUNKING", "Chunking"), ("READY", "Ready"), ("FAILED", "Failed"), ("DELETING", "Deleting")], default="UPLOADED", max_length=20)),
                ("processing_error", models.TextField(blank=True)),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
                ("indexed_at", models.DateTimeField(blank=True, null=True)),
                ("owner", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="documents", to=settings.AUTH_USER_MODEL)),
                ("workspace", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="documents", to="workspaces.workspace")),
            ],
            options={"ordering": ["-uploaded_at"]},
        ),
        migrations.CreateModel(
            name="DocumentChunk",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("page_number", models.PositiveIntegerField()),
                ("chunk_index", models.PositiveIntegerField()),
                ("section_heading", models.CharField(blank=True, max_length=255)),
                ("content", models.TextField()),
                ("content_hash", models.CharField(db_index=True, max_length=64)),
                ("token_count", models.PositiveIntegerField(default=0)),
                ("start_offset", models.PositiveIntegerField(default=0)),
                ("end_offset", models.PositiveIntegerField(default=0)),
                ("embedding", models.JSONField(blank=True, help_text="Local fallback. Replace with pgvector VectorField in production migration.", null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("document", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="chunks", to="documents.document")),
                ("owner", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="document_chunks", to=settings.AUTH_USER_MODEL)),
                ("workspace", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="chunks", to="workspaces.workspace")),
            ],
            options={"ordering": ["document_id", "page_number", "chunk_index"]},
        ),
        migrations.AddConstraint(
            model_name="document",
            constraint=models.UniqueConstraint(fields=("owner", "file_hash"), name="unique_document_per_owner"),
        ),
        migrations.AddConstraint(
            model_name="documentchunk",
            constraint=models.UniqueConstraint(fields=("document", "page_number", "chunk_index", "content_hash"), name="unique_document_chunk"),
        ),
    ]
