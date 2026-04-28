"""
File-handling utilities: content-disposition header, MIME helpers.

Pure functions — no app.state or DB dependencies.
"""


def _content_disposition(disposition: str, filename: str) -> str:
    """Build a Content-Disposition header value that handles non-ASCII filenames.

    Uses RFC 5987 filename* parameter for filenames containing characters
    outside latin-1 (e.g. em dashes, accented letters, CJK), which would
    otherwise cause a UnicodeEncodeError when Starlette encodes the header.
    """
    try:
        filename.encode("latin-1")
        return f'{disposition}; filename="{filename}"'
    except (UnicodeEncodeError, UnicodeDecodeError):
        from urllib.parse import quote as _quote
        return f"{disposition}; filename*=UTF-8''{_quote(filename, safe='')}"
