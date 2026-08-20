import hashlib
import fitz
from django.db import transaction
from django.utils import timezone
from .chunking import chunk_page
from .models import Document, DocumentChunk
from .storage import get_document_storage
from .visual_equations import extract_visual_equation_chunks

@transaction.atomic
def index_document(document: Document, pdf_bytes: bytes) -> Document:
    document.processing_status = Document.Status.EXTRACTING
    document.save(update_fields=["processing_status"])
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        document.page_count = pdf.page_count
        document.processing_status = Document.Status.CHUNKING
        document.save(update_fields=["page_count", "processing_status"])
        # Re-indexing must replace the old chunk set; otherwise a changed
        # chunking strategy leaves stale and duplicate passages searchable.
        document.chunks.all().delete()
        pending = []
        for page_idx in range(pdf.page_count):
            page = pdf.load_page(page_idx)
            text = page.get_text("text")
            page_chunks = list(chunk_page(text))
            page_chunks.extend(
                extract_visual_equation_chunks(
                    page,
                    page_number=page_idx + 1,
                    page_text=text,
                )
            )

            for chunk_idx, content in enumerate(page_chunks):
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
    previous_status = document.processing_status

    document.processing_status = Document.Status.DELETING
    document.save(update_fields=["processing_status"])

    try:
        get_document_storage().delete(document.storage_path)
    except Exception as exc:
        # A failed remote delete must not leave the document permanently
        # stuck in the DELETING state.
        document.processing_status = (
            previous_status
            if previous_status != Document.Status.DELETING
            else Document.Status.FAILED
        )
        document.processing_error = str(exc)[:1000]
        document.save(
            update_fields=[
                "processing_status",
                "processing_error",
            ]
        )
        raise

    document.delete()
