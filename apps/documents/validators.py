import hashlib
import fitz
from django.conf import settings
from django.core.exceptions import ValidationError

def validate_pdf(uploaded_file):
    if not uploaded_file.name.lower().endswith(".pdf"):
        raise ValidationError("Only PDF files are allowed.")
    if uploaded_file.size > settings.MAX_PDF_SIZE_MB * 1024 * 1024:
        raise ValidationError(f"PDF must be {settings.MAX_PDF_SIZE_MB} MB or smaller.")
    data = uploaded_file.read()
    uploaded_file.seek(0)
    if not data.startswith(b"%PDF-"):
        raise ValidationError("The uploaded file is not a valid PDF.")
    try:
        pdf = fitz.open(stream=data, filetype="pdf")
        if pdf.needs_pass:
            raise ValidationError("Password-protected PDFs are not supported yet.")
        if pdf.page_count == 0:
            raise ValidationError("The PDF does not contain any pages.")
        if pdf.page_count > settings.MAX_PDF_PAGES:
            raise ValidationError(f"PDF must contain no more than {settings.MAX_PDF_PAGES} pages.")
        page_count = pdf.page_count
        pdf.close()
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError("The PDF is corrupted or unreadable.") from exc
    return {"bytes": data, "sha256": hashlib.sha256(data).hexdigest(), "page_count": page_count}
