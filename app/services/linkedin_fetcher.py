import re
import asyncio
from typing import Optional
import requests
from bs4 import BeautifulSoup


class LinkedInFetcher:
    """Service for fetching and extracting text content from a LinkedIn public profile URL.

    Note: LinkedIn may block automated scraping. This fetcher works best with public profiles
    and is designed as a best-effort extractor. If content cannot be retrieved, it returns
    an empty string so the caller can handle fallbacks.
    """

    def __init__(self, request_timeout_seconds: int = 20):
        self.request_timeout_seconds = request_timeout_seconds
        self._headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

    async def fetch_profile_text(self, url: str) -> str:
        return await asyncio.to_thread(self._fetch_profile_text_sync, url)

    def _fetch_profile_text_sync(self, url: str) -> str:
        try:
            if not url or "linkedin.com" not in url:
                return ""

            response = requests.get(url, headers=self._headers, timeout=self.request_timeout_seconds)
            if response.status_code != 200:
                return ""

            html = response.text
            # Quick block/redirect detection
            if any(block in html.lower() for block in ["signin", "login", "captcha", "enable javascript"]):
                # Likely blocked or requires auth/JS
                return ""

            soup = BeautifulSoup(html, "html.parser")

            extracted_sections = []

            # Title and meta description
            page_title = (soup.title.string or "").strip() if soup.title else ""
            if page_title:
                extracted_sections.append(f"Title: {page_title}")

            og_desc = soup.find("meta", attrs={"property": "og:description"})
            if og_desc and og_desc.get("content"):
                extracted_sections.append(f"About: {og_desc['content'].strip()}")

            # Visible text strategy: pick text from sections that resemble profile blocks
            # This is heuristic and may evolve.
            candidates = []
            for selector in [
                "section",
                "main",
                "div",
            ]:
                candidates.extend(soup.select(selector))

            def score_section(node) -> int:
                text = node.get_text(separator=" ", strip=True)
                score = 0
                keywords = [
                    "experience",
                    "education",
                    "skills",
                    "certification",
                    "projects",
                    "about",
                    "summary",
                    "activity",
                    "languages",
                    "honors",
                    "awards",
                    "volunteer",
                ]
                lower = text.lower()
                for kw in keywords:
                    if kw in lower:
                        score += 1
                score += min(len(text) // 200, 5)
                return score

            # Take top N sections by heuristic score
            candidates = sorted(candidates, key=score_section, reverse=True)[:12]

            for node in candidates:
                text = node.get_text(separator=" ", strip=True)
                text = self._clean_text(text)
                if len(text) > 200:
                    extracted_sections.append(text)

            combined = "\n\n".join(self._dedupe_preserve_order(extracted_sections))
            # Keep a sane limit to avoid overlong prompts
            return combined[:30000]
        except Exception:
            return ""

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        # Remove repeated UI artifacts
        text = re.sub(r"See more|Show more|See less|Show less", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        seen = set()
        result: list[str] = []
        for item in items:
            key = item[:200]
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result 