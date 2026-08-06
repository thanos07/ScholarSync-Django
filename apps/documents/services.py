import hashlib
import fitz
from django.db import transaction
from django.utils import timezone
from .chunking import chunk_page
from .models import Document, DocumentChunk
from .storage import get_document_storage

@transaction.atomic
def index_document(document: Document, pdf_bytes: bytes) -> Document:
    document.processing_status = Document.Status.EXTRACTING
    document.save(update_fields=["processing_status"])
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        document.page_count = pdf.page_count
        document.processing_status = Document.Status.CHUNKING
        document.save(update_fields=["page_count", "processing_status"])
        pending = []
        for page_idx in range(pdf.page_count):
            text = pdf.load_page(page_idx).get_text("text")
            for chunk_idx, content in enumerate(chunk_page(text)):
                pending.append(DocumentChunk(
                    document=document, workspace=document.workspace, owner=document.owner,
                    page_number=page_idx + 1, chunk_index=chunk_idx,
                    content=content, content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    token_count=max(1, len(content.split())), start_offset=0, end_offset=len(content),
                ))
        pdf.close()
        DocumentChunk.objects.bulk_create(pending, ignore_conflicts=True)
        document.processing_status = Document.Status.READY
        document.indexed_at = timezone.now()
        document.processing_error = ""
        document.save(update_fields=["processing_status", "indexed_at", "processing_error"])
    except Exception as exc:
        document.processing_status = Document.Status.FAILED
        document.processing_error = str(exc)[:1000]
        document.save(update_fields=["processing_status", "processing_error"])
        raise
    return document

def remove_document(document: Document):
    document.processing_status = Document.Status.DELETING
    document.save(update_fields=["processing_status"])
    get_document_storage().delete(document.storage_path)
    document.delete()
