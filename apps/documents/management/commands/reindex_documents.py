from django.core.management.base import BaseCommand, CommandError

from apps.documents.models import Document
from apps.documents.services import index_document
from apps.documents.storage import get_document_storage


class Command(BaseCommand):
    help = "Rebuild page chunks for already uploaded private-workspace PDFs."

    def add_arguments(self, parser):
        parser.add_argument(
            "--document-id",
            help="Re-index only one document UUID. By default all documents are rebuilt.",
        )

    def handle(self, *args, **options):
        documents = Document.objects.select_related("owner", "workspace").all()
        document_id = options.get("document_id")
        if document_id:
            documents = documents.filter(id=document_id)
        if not documents.exists():
            raise CommandError("No matching document was found.")

        storage = get_document_storage()
        total = documents.count()
        for index, document in enumerate(documents.iterator(), 1):
            self.stdout.write(
                f"[{index}/{total}] Re-indexing {document.display_title} ({document.id})..."
            )
            pdf_bytes = storage.read(document.storage_path)
            index_document(document, pdf_bytes)
            self.stdout.write(
                self.style.SUCCESS(
                    f"  ready: {document.chunks.count()} chunks across {document.page_count} pages"
                )
            )
