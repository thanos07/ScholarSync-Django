from pathlib import Path
from urllib.parse import quote
import httpx
from django.conf import settings

class LocalPrivateStorage:
    def save(self, path: str, data: bytes) -> str:
        target = settings.MEDIA_ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return path
    def read(self, path: str) -> bytes:
        return (settings.MEDIA_ROOT / path).read_bytes()
    def delete(self, path: str) -> None:
        (settings.MEDIA_ROOT / path).unlink(missing_ok=True)

class SupabasePrivateStorage:
    def __init__(self):
        self.base = settings.SUPABASE_URL.rstrip("/")
        self.key = settings.SUPABASE_SERVICE_ROLE_KEY
        self.bucket = settings.SUPABASE_STORAGE_BUCKET
        if not self.base or not self.key:
            raise RuntimeError("Supabase storage environment variables are missing.")
    @property
    def headers(self):
        return {"Authorization": f"Bearer {self.key}", "apikey": self.key}
    def save(self, path: str, data: bytes) -> str:
        url = f"{self.base}/storage/v1/object/{self.bucket}/{quote(path)}"
        response = httpx.post(url, headers={**self.headers, "Content-Type": "application/pdf", "x-upsert": "false"}, content=data, timeout=60)
        response.raise_for_status()
        return path
    def read(self, path: str) -> bytes:
        url = f"{self.base}/storage/v1/object/authenticated/{self.bucket}/{quote(path)}"
        response = httpx.get(url, headers=self.headers, timeout=60)
        response.raise_for_status()
        return response.content
    def delete(self, path: str) -> None:
        url = f"{self.base}/storage/v1/object/{self.bucket}"
        response = httpx.delete(url, headers={**self.headers, "Content-Type": "application/json"}, json={"prefixes": [path]}, timeout=30)
        response.raise_for_status()

def get_document_storage():
    if settings.DOCUMENT_STORAGE_BACKEND == "supabase":
        return SupabasePrivateStorage()
    return LocalPrivateStorage()
