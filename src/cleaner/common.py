from pathlib import Path
from pydantic import BaseModel
from typing import NamedTuple, Optional, cast

import concurrent.futures

class Document(BaseModel):
    """Class for storing a piece of text and associated metadata."""

    page_content: str

    vector: Optional[list[float]] = None

    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """
    metadata: dict = {}
    
class FileEncoding(NamedTuple):
    """A file encoding as the NamedTuple."""

    encoding: Optional[str]
    """The encoding of the file."""
    confidence: float
    """The confidence of the encoding."""
    language: Optional[str]
    """The language of the file."""


def detect_file_encodings(file_path: str, timeout: int = 5) -> list[FileEncoding]:
    """Try to detect the file encoding.

    Returns a list of `FileEncoding` tuples with the detected encodings ordered
    by confidence.

    Args:
        file_path: The path to the file to detect the encoding for.
        timeout: The timeout in seconds for the encoding detection.
    """
    import chardet

    def read_and_detect(file_path: str) -> list[dict]:
        rawdata = Path(file_path).read_bytes()
        return cast(list[dict], chardet.detect_all(rawdata))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(read_and_detect, file_path)
        try:
            encodings = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Timeout reached while detecting encoding for {file_path}")

    if all(encoding["encoding"] is None for encoding in encodings):
        raise RuntimeError(f"Could not detect encoding for {file_path}")
    return [FileEncoding(**enc) for enc in encodings if enc["encoding"] is not None]
