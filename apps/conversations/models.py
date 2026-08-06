import uuid
from django.conf import settings
from django.db import models

class Conversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workspace = models.ForeignKey("workspaces.Workspace", on_delete=models.CASCADE, related_name="conversations")
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="conversations")
    title = models.CharField(max_length=180, default="New research conversation")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    class Meta:
        ordering = ["-updated_at"]

class Message(models.Model):
    class Role(models.TextChoices):
        USER = "USER", "User"
        ASSISTANT = "ASSISTANT", "ScholarSync"
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=12, choices=Role.choices)
    content = models.TextField()
    model_name = models.CharField(max_length=100, blank=True)
    confidence = models.CharField(max_length=20, blank=True)
    response_time_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    class Meta:
        ordering = ["created_at"]

class Citation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name="citations")
    chunk = models.ForeignKey("documents.DocumentChunk", on_delete=models.CASCADE, related_name="citations")
    document = models.ForeignKey("documents.Document", on_delete=models.CASCADE, related_name="citations")
    page_number = models.PositiveIntegerField()
    citation_number = models.PositiveIntegerField()
    quoted_passage = models.TextField()
    retrieval_score = models.FloatField(default=0)
    verification_status = models.CharField(max_length=20, default="VALID")
    class Meta:
        ordering = ["citation_number"]
