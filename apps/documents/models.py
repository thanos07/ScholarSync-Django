import uuid
from django.conf import settings
from django.db import models

class Document(models.Model):
    class Status(models.TextChoices):
        UPLOADED = "UPLOADED", "Uploaded"
        EXTRACTING = "EXTRACTING", "Extracting"
        CHUNKING = "CHUNKING", "Chunking"
        READY = "READY", "Ready"
        FAILED = "FAILED", "Failed"
        DELETING = "DELETING", "Deleting"
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="documents")
    workspace = models.ForeignKey("workspaces.Workspace", on_delete=models.CASCADE, related_name="documents")
    original_filename = models.CharField(max_length=255)
    display_title = models.CharField(max_length=255)
    storage_path = models.CharField(max_length=500)
    file_hash = models.CharField(max_length=64, db_index=True)
    file_size = models.PositiveBigIntegerField(default=0)
    page_count = models.PositiveIntegerField(default=0)
    mime_type = models.CharField(max_length=100, default="application/pdf")
    processing_status = models.CharField(max_length=20, choices=Status.choices, default=Status.UPLOADED)
    processing_error = models.TextField(blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    indexed_at = models.DateTimeField(null=True, blank=True)
    class Meta:
        constraints = [models.UniqueConstraint(fields=["owner", "file_hash"], name="unique_document_per_owner")]
        ordering = ["-uploaded_at"]
    def __str__(self):
        return self.display_title

class DocumentChunk(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    workspace = models.ForeignKey("workspaces.Workspace", on_delete=models.CASCADE, related_name="chunks")
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="document_chunks")
    page_number = models.PositiveIntegerField()
    chunk_index = models.PositiveIntegerField()
    section_heading = models.CharField(max_length=255, blank=True)
    content = models.TextField()
    content_hash = models.CharField(max_length=64, db_index=True)
    token_count = models.PositiveIntegerField(default=0)
    start_offset = models.PositiveIntegerField(default=0)
    end_offset = models.PositiveIntegerField(default=0)
    embedding = models.JSONField(null=True, blank=True, help_text="Local fallback. Replace with pgvector VectorField in production migration.")
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        constraints = [models.UniqueConstraint(fields=["document", "page_number", "chunk_index", "content_hash"], name="unique_document_chunk")]
        ordering = ["document_id", "page_number", "chunk_index"]
