"""Abstract interface for document loader implementations."""

from collections.abc import Iterator
from typing import Optional, cast


from cleaner.blob import Blob
from cleaner.common import Document



class PdfExtractor():
    """Load pdf files.


    Args:
        file_path: Path to the file to load.
    """

    def __init__(self, file_path: str, file_cache_key: Optional[str] = None):
        """Initialize with file path."""
        self._file_path = file_path

    def extract(self) -> list[Document]:
        plaintext_file_exists = False
        documents = list(self.load())
        text_list = []
        for document in documents:
            text_list.append(document.page_content)
        text = "\n\n".join(text_list)
        return documents
    
    def extract_text(self):
        text_list = []
        for document in self.load():
            text_list.append(document.page_content)
        text = "\n\n".join(text_list)
        return text

    def load(
        self,
    ) -> Iterator[Document]:
        """Lazy load given path as pages."""
        blob = Blob.from_path(self._file_path)
        yield from self.parse(blob)

    def parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdfium2  # type: ignore

        with blob.as_bytes_io() as file_path:
            pdf_reader = pypdfium2.PdfDocument(file_path, autoclose=True)
            try:
                for page_number, page in enumerate(pdf_reader):
                    text_page = page.get_textpage()
                    content = text_page.get_text_range()
                    text_page.close()
                    page.close()
                    metadata = {"source": blob.source, "page": page_number}
                    yield Document(page_content=content, metadata=metadata)
            finally:
                pdf_reader.close()


if __name__ == "__main__":
    pdf_extractor = PdfExtractor("/home/kaisa/Desktop/Project/Github/advance-rag/src/dataset/test.pdf")
    docs = pdf_extractor.extract()
    print(docs)