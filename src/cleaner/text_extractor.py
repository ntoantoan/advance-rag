"""Abstract interface for document loader implementations."""

from pathlib import Path
from typing import Optional


from typing import NamedTuple, Optional, cast
from cleaner.common import Document
from cleaner.common import detect_file_encodings




class TextExtractor():
    """Load text files.


    Args:
        file_path: Path to the file to load.
    """

    def __init__(self, file_path: str, encoding: Optional[str] = None, autodetect_encoding: bool = False):
        """Initialize with file path."""
        self._file_path = file_path
        self._encoding = encoding
        self._autodetect_encoding = autodetect_encoding

    def extract(self) -> list[Document]:
        """Load from file path."""
        text = ""
        try:
            text = Path(self._file_path).read_text(encoding=self._encoding)
        except UnicodeDecodeError as e:
            if self._autodetect_encoding:
                detected_encodings = detect_file_encodings(self._file_path)
                for encoding in detected_encodings:
                    try:
                        text = Path(self._file_path).read_text(encoding=encoding.encoding)
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading {self._file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading {self._file_path}") from e

        metadata = {"source": self._file_path}
        return [Document(page_content=text, metadata=metadata)]


if __name__ == "__main__":
    text_extractor = TextExtractor("/home/kaisa/Desktop/Project/Github/advance-rag/src/dataset/test.txt")
    docs = text_extractor.extract()
    print(docs)