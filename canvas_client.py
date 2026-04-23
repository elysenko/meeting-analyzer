"""Canvas LMS API client for fetching courses, assignments, files, and modules."""
import logging
from typing import Any
from urllib.parse import urljoin

import httpx

logger = logging.getLogger("meeting-analyzer")


class CanvasClient:
    """Async client for Canvas LMS REST API.

    Uses personal access tokens for authentication.
    API documentation: https://canvas.instructure.com/doc/api/
    """

    def __init__(self, instance_url: str, access_token: str):
        """Initialize the Canvas client.

        Args:
            instance_url: Base URL of Canvas instance (e.g., https://byu.instructure.com)
            access_token: Personal access token from Canvas settings
        """
        self.instance_url = instance_url.rstrip("/")
        self.access_token = access_token
        self.api_base = f"{self.instance_url}/api/v1"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

    async def _get(self, endpoint: str, params: dict | None = None) -> Any:
        """Make a GET request to the Canvas API."""
        url = f"{self.api_base}{endpoint}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, headers=self._headers(), params=params or {})
            if resp.status_code == 401:
                raise CanvasAuthError("Invalid or expired Canvas token")
            if resp.status_code == 403:
                raise CanvasPermissionError("Access denied to this resource")
            if resp.status_code == 404:
                raise CanvasNotFoundError(f"Resource not found: {endpoint}")
            resp.raise_for_status()
            return resp.json()

    async def _get_paginated(self, endpoint: str, params: dict | None = None, limit: int = 100) -> list[Any]:
        """Fetch all pages of a paginated endpoint."""
        results = []
        url = f"{self.api_base}{endpoint}"
        params = params or {}
        params["per_page"] = min(limit, 100)

        async with httpx.AsyncClient(timeout=30.0) as client:
            while url and len(results) < limit:
                resp = await client.get(url, headers=self._headers(), params=params)
                if resp.status_code == 401:
                    raise CanvasAuthError("Invalid or expired Canvas token")
                if resp.status_code == 403:
                    raise CanvasPermissionError("Access denied to this resource")
                resp.raise_for_status()

                data = resp.json()
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)

                # Check for next page via Link header
                links = resp.headers.get("Link", "")
                url = None
                for link in links.split(","):
                    if 'rel="next"' in link:
                        url = link.split(";")[0].strip().strip("<>")
                        params = {}  # Params are encoded in the next URL
                        break

        return results[:limit]

    async def validate_token(self) -> dict:
        """Validate the access token by fetching the current user profile.

        Returns:
            User profile dict with id, name, etc.

        Raises:
            CanvasAuthError: If token is invalid
        """
        return await self._get("/users/self")

    async def get_courses(self, enrollment_state: str = "active") -> list[dict]:
        """List courses the user is enrolled in.

        Args:
            enrollment_state: Filter by enrollment state (active, completed, invited)

        Returns:
            List of course dicts with id, name, course_code, etc.
        """
        return await self._get_paginated(
            "/courses",
            params={
                "enrollment_state": enrollment_state,
                "include[]": ["term", "total_students"],
            },
        )

    async def get_course(self, course_id: int) -> dict:
        """Get details for a specific course.

        Args:
            course_id: Canvas course ID

        Returns:
            Course dict with id, name, syllabus_body if included
        """
        return await self._get(
            f"/courses/{course_id}",
            params={"include[]": ["syllabus_body", "term"]},
        )

    async def get_assignments(self, course_id: int) -> list[dict]:
        """List all assignments for a course.

        Args:
            course_id: Canvas course ID

        Returns:
            List of assignment dicts with id, name, due_at, points_possible, etc.
        """
        return await self._get_paginated(
            f"/courses/{course_id}/assignments",
            params={
                "order_by": "due_at",
                "include[]": ["submission"],
            },
        )

    async def get_files(self, course_id: int) -> list[dict]:
        """List all files in a course.

        Args:
            course_id: Canvas course ID

        Returns:
            List of file dicts with id, display_name, size, content-type, url, etc.
        """
        return await self._get_paginated(
            f"/courses/{course_id}/files",
            params={"sort": "updated_at", "order": "desc"},
        )

    async def get_file(self, file_id: int) -> dict:
        """Get metadata for a specific file.

        Args:
            file_id: Canvas file ID

        Returns:
            File dict with id, display_name, url (download URL), size, etc.
        """
        return await self._get(f"/files/{file_id}")

    async def download_file(self, file_url: str) -> bytes:
        """Download file content from a Canvas file URL.

        Args:
            file_url: The URL from the file's 'url' field (redirects to actual content)

        Returns:
            File content as bytes
        """
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            resp = await client.get(file_url, headers=self._headers())
            resp.raise_for_status()
            return resp.content

    async def get_modules(self, course_id: int) -> list[dict]:
        """List all modules for a course.

        Args:
            course_id: Canvas course ID

        Returns:
            List of module dicts with id, name, items_count, etc.
        """
        return await self._get_paginated(
            f"/courses/{course_id}/modules",
            params={"include[]": ["items"]},
        )

    async def get_module_items(self, course_id: int, module_id: int) -> list[dict]:
        """List items in a module.

        Args:
            course_id: Canvas course ID
            module_id: Module ID

        Returns:
            List of module item dicts with type, title, content_id, etc.
        """
        return await self._get_paginated(
            f"/courses/{course_id}/modules/{module_id}/items",
            params={"include[]": ["content_details"]},
        )

    async def get_pages(self, course_id: int) -> list[dict]:
        """List all pages in a course.

        Args:
            course_id: Canvas course ID

        Returns:
            List of page dicts with url, title, updated_at, etc.
        """
        return await self._get_paginated(f"/courses/{course_id}/pages")

    async def get_page(self, course_id: int, page_url: str) -> dict:
        """Get a specific page's content.

        Args:
            course_id: Canvas course ID
            page_url: The page's URL identifier (from pages list)

        Returns:
            Page dict with title, body (HTML content), etc.
        """
        return await self._get(f"/courses/{course_id}/pages/{page_url}")


class CanvasError(Exception):
    """Base exception for Canvas API errors."""
    pass


class CanvasAuthError(CanvasError):
    """Raised when authentication fails."""
    pass


class CanvasPermissionError(CanvasError):
    """Raised when user lacks permission to access a resource."""
    pass


class CanvasNotFoundError(CanvasError):
    """Raised when a resource is not found."""
    pass
