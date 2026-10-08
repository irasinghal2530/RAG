#!/usr/bin/env python3

"""
download_corpus_v3.py

Kaeser RAG Corpus Downloader
----------------------------

Purpose:
    Automatically discover dynamically-loaded Kaeser PDF/document links
    using a real Chromium browser, match them against the corpus manifest,
    and download/validate the documents.

Expected files:
    kaeser_corpus_manifest.csv

Output:
    corpus_pdfs/
    kaeser_corpus_manifest.updated.csv
    kaeser_corpus_manifest.updated.json
    browser_discovered_links.json
    download_log.txt

Install:
    pip install playwright
    playwright install chromium

Usage:

    # Discover links + download
    python download_corpus_v3.py

    # Only discover links, don't download
    python download_corpus_v3.py --discover-only

    # Run only one document
    python download_corpus_v3.py --only PRD-001

    # Run visibly so you can watch the browser
    python download_corpus_v3.py --headed

    # Slower / safer
    python download_corpus_v3.py --delay 3

"""

import argparse
import asyncio
import csv
import datetime
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright


# ============================================================
# CONFIG
# ============================================================

MANIFEST = Path("kaeser_corpus_manifest.csv")

OUT_DIR = Path("corpus_pdfs")

LINKS_FILE = Path("browser_discovered_links.json")

UPDATED_MANIFEST = Path(
    "kaeser_corpus_manifest.updated.csv"
)

UPDATED_JSON = Path(
    "kaeser_corpus_manifest.updated.json"
)

LOG_FILE = Path("download_log.txt")

KAESER_DOWNLOADS_URLS = [
    "https://us.kaeser.com/compressed-air-resources/downloads/",
    "https://pr.kaeser.com/en/compressed-air-resources/downloads/",
]

# Words which should not influence matching much.
STOP_WORDS = {
    "series",
    "product",
    "products",
    "literature",
    "the",
    "and",
    "of",
    "for",
    "to",
    "in",
    "a",
    "an",
    "with",
    "air",
    "compressed",
}

MIN_MATCH_SCORE = 2

DEFAULT_DELAY = 1.5

MAX_FILE_SIZE = 100 * 1024 * 1024


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("kaeser")


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value)

    value = value.replace("_", " ")
    value = value.replace("-", " ")

    return re.sub(
        r"\s+",
        " ",
        value.lower()
    ).strip()


def tokens(value):
    """
    Convert text into matching tokens.
    """

    text = normalize_text(value)

    raw = re.findall(
        r"[a-z0-9]+(?:\.[0-9]+)?",
        text
    )

    return {
        token
        for token in raw
        if (
            token not in STOP_WORDS
            and len(token) > 1
        )
    }


# ============================================================
# MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    with MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as fh:

        rows = list(
            csv.DictReader(fh)
        )

    if not rows:
        raise ValueError(
            "Manifest is empty."
        )

    required = {
        "document_id",
        "title",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"Manifest missing columns: {missing}"
        )

    # Ensure columns exist.
    extra_columns = [
        "downloaded",
        "sha256",
        "local_path",
        "retrieved_at",
        "resolved_file_url",
        "download_error",
        "match_score",
    ]

    for row in rows:
        for column in extra_columns:
            row.setdefault(column, "")

    return rows


# ============================================================
# URL FILTERING
# ============================================================

def looks_like_document_url(url):

    if not url:
        return False

    url_lower = url.lower()

    if url_lower.startswith(
        (
            "javascript:",
            "mailto:",
            "tel:",
            "#",
        )
    ):
        return False

    # Direct PDFs
    if ".pdf" in url_lower:
        return True

    # Common download endpoints
    if "download" in url_lower:
        return True

    if "/media/" in url_lower:
        return True

    if "document" in url_lower:
        return True

    return False


# ============================================================
# BROWSER LINK DISCOVERY
# ============================================================

async def discover_links(page, url):

    logger.info(
        "Opening: %s",
        url
    )

    try:

        response = await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=60000,
        )

        if response:
            logger.info(
                "HTTP status: %s",
                response.status
            )

    except Exception as exc:

        logger.warning(
            "Initial page load problem: %s",
            exc
        )

    # Allow JS to execute.
    await page.wait_for_timeout(5000)

    # Scroll to force lazy loading.
    for _ in range(5):

        await page.mouse.wheel(
            0,
            1200
        )

        await page.wait_for_timeout(
            1000
        )

    # --------------------------------------------------------
    # Collect anchors
    # --------------------------------------------------------

    anchors = await page.locator(
        "a[href]"
    ).evaluate_all(
        """
        elements => elements.map(a => ({
            text: (a.innerText || a.textContent || '').trim(),
            href: a.href
        }))
        """
    )

    discovered = []

    for item in anchors:

        href = item.get("href")

        if not href:
            continue

        if not looks_like_document_url(href):
            continue

        discovered.append(
            {
                "text": item.get("text", ""),
                "href": href,
                "source_page": url,
                "discovery_method": "anchor",
            }
        )

    logger.info(
        "Found %d document-like anchors",
        len(discovered)
    )

    # --------------------------------------------------------
    # Inspect page HTML for PDF URLs
    # --------------------------------------------------------

    html = await page.content()

    pdf_urls = set(
        re.findall(
            r'https?://[^"\']+?\.pdf(?:\?[^"\']*)?',
            html,
            flags=re.I,
        )
    )

    for pdf_url in pdf_urls:

        discovered.append(
            {
                "text": "",
                "href": pdf_url,
                "source_page": url,
                "discovery_method": "html",
            }
        )

    logger.info(
        "Found %d PDFs embedded in HTML",
        len(pdf_urls)
    )

    return discovered


# ============================================================
# NETWORK DISCOVERY
# ============================================================

async def discover_network_requests(browser, url):

    """
    Load page and record requests which look like PDFs/documents.

    This is useful when the PDF URL is generated dynamically.
    """

    requests_found = []

    context = await browser.new_context(
        accept_downloads=True
    )

    page = await context.new_page()

    async def on_response(response):

        try:

            response_url = response.url

            content_type = (
                response.headers.get(
                    "content-type",
                    ""
                )
                .lower()
            )

            if (
                "application/pdf" in content_type
                or ".pdf" in response_url.lower()
            ):

                requests_found.append(
                    {
                        "text": "",
                        "href": response_url,
                        "source_page": url,
                        "discovery_method": "network",
                        "content_type": content_type,
                    }
                )

        except Exception:
            pass

    page.on(
        "response",
        on_response
    )

    try:

        await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=60000,
        )

    except Exception as exc:

        logger.warning(
            "Network discovery page load: %s",
            exc
        )

    await page.wait_for_timeout(
        7000
    )

    # Trigger lazy content.
    for _ in range(8):

        await page.mouse.wheel(
            0,
            1500
        )

        await page.wait_for_timeout(
            800
        )

    await context.close()

    logger.info(
        "Network discovery found %d document requests",
        len(requests_found)
    )

    return requests_found


# ============================================================
# DEDUPLICATE LINKS
# ============================================================

def deduplicate_links(links):

    unique = {}

    for link in links:

        href = link.get("href")

        if not href:
            continue

        # Strip fragments.
        href = href.split("#")[0]

        if href not in unique:

            link = dict(link)
            link["href"] = href

            unique[href] = link

        else:

            # Preserve useful text if first record had none.
            if (
                not unique[href].get("text")
                and link.get("text")
            ):
                unique[href]["text"] = link["text"]

    return list(
        unique.values()
    )


# ============================================================
# MATCHING
# ============================================================

def build_manifest_text(row):

    fields = [
        row.get("title", ""),
        row.get("product_family", ""),
        row.get("category", ""),
        row.get("document_type", ""),
        row.get("notes", ""),
    ]

    return " ".join(
        str(x)
        for x in fields
        if x
    )


def score_link(row, link):

    manifest_tokens = tokens(
        build_manifest_text(row)
    )

    url = link.get("href", "")

    filename = Path(
        urlparse(url).path
    ).name

    link_tokens = (
        tokens(link.get("text", ""))
        |
        tokens(filename)
    )

    if not manifest_tokens or not link_tokens:
        return 0

    overlap = (
        manifest_tokens
        &
        link_tokens
    )

    score = len(overlap)

    # Strong bonus for exact product identifiers.
    title = normalize_text(
        row.get("title", "")
    )

    filename_text = normalize_text(
        filename
    )

    product_family = normalize_text(
        row.get("product_family", "")
    )

    if (
        product_family
        and product_family in filename_text
    ):
        score += 4

    # Match model identifiers such as SFC 18-45S.
    identifiers = re.findall(
        r"\b[a-z]{2,5}[\s\-]?\d+(?:[\s\-]?\d+)?[a-z]?\b",
        title,
        flags=re.I,
    )

    for identifier in identifiers:

        identifier_clean = (
            identifier
            .replace(" ", "")
            .replace("-", "")
            .lower()
        )

        filename_clean = (
            filename_text
            .replace(" ", "")
            .replace("-", "")
        )

        if (
            identifier_clean
            and identifier_clean in filename_clean
        ):
            score += 6

    return score


def match_document(row, links):

    scored = []

    for link in links:

        score = score_link(
            row,
            link
        )

        if score <= 0:
            continue

        scored.append(
            (
                score,
                link
            )
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    if not scored:
        return None, []

    best_score, best_link = scored[0]

    # Don't blindly download weak matches.
    if best_score < MIN_MATCH_SCORE:

        return None, scored[:5]

    # Ambiguous match.
    if len(scored) > 1:

        second_score = scored[1][0]

        if (
            second_score == best_score
            and scored[1][1]["href"]
            != best_link["href"]
        ):

            return None, scored[:5]

    return best_link, scored[:5]


# ============================================================
# PDF VALIDATION
# ============================================================

def validate_pdf(path):

    if not path.exists():
        raise ValueError(
            "File does not exist."
        )

    size = path.stat().st_size

    if size == 0:
        raise ValueError(
            "Downloaded file is empty."
        )

    if size > MAX_FILE_SIZE:
        raise ValueError(
            f"File exceeds {MAX_FILE_SIZE} bytes."
        )

    with path.open(
        "rb"
    ) as fh:

        header = fh.read(8)

    if not header.startswith(
        b"%PDF"
    ):
        raise ValueError(
            f"Not a PDF. Header={header!r}"
        )

    return size


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as fh:

        while True:

            chunk = fh.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# DOWNLOAD USING PLAYWRIGHT
# ============================================================

async def download_pdf(
    context,
    url,
    destination,
):

    page = await context.new_page()

    # Registered before navigation so a file download is not missed.
    # download.ashx links never become a page response.
    download_task = asyncio.create_task(
        page.wait_for_event(
            "download",
            timeout=90000,
        )
    )

    try:

        logger.info(
            "Downloading: %s",
            url
        )

        navigation_error = None

        response = None

        try:

            response = await page.goto(
                url,
                wait_until="commit",
                timeout=90000,
            )

        except Exception as exc:

            if "Download is starting" not in str(exc):
                raise

            navigation_error = exc

        if (
            navigation_error is not None
            or download_task.done()
        ):

            download = await download_task

            failure = await download.failure()

            if failure:

                raise RuntimeError(
                    failure
                )

            await download.save_as(
                destination
            )

            logger.info(
                "Saved browser download: %s",
                destination.name
            )

            return destination

        if response:

            status = response.status

            logger.info(
                "PDF response HTTP %s",
                status
            )

            if status >= 400:

                raise RuntimeError(
                    f"HTTP {status}"
                )

            body = await response.body()

        else:

            raise RuntimeError(
                "No response received."
            )

        if not body.startswith(
            b"%PDF"
        ):

            content_type = ""

            if response:
                content_type = (
                    response.headers.get(
                        "content-type",
                        ""
                    )
                )

            raise ValueError(
                "Response is not a PDF. "
                f"content-type={content_type}"
            )

        destination.write_bytes(
            body
        )

        return destination

    finally:

        if not download_task.done():
            download_task.cancel()

        try:
            await download_task
        except (asyncio.CancelledError, Exception):
            pass

        await page.close()


# ============================================================
# SAVE DISCOVERED LINKS
# ============================================================

def save_links(links):

    LINKS_FILE.write_text(
        json.dumps(
            links,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Saved %d discovered links -> %s",
        len(links),
        LINKS_FILE
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

def save_manifest(rows):

    fieldnames = list(
        rows[0].keys()
    )

    with UPDATED_MANIFEST.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    UPDATED_JSON.write_text(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    parser = argparse.ArgumentParser(
        description="Kaeser browser-based corpus downloader"
    )

    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show Chromium browser"
    )

    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover links; don't download"
    )

    parser.add_argument(
        "--only",
        help="Process only one document_id"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Delay between downloads"
    )

    args = parser.parse_args()

    OUT_DIR.mkdir(
        exist_ok=True
    )

    rows = load_manifest()

    logger.info(
        "Loaded %d manifest documents",
        len(rows)
    )

    # --------------------------------------------------------
    # Browser
    # --------------------------------------------------------

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=not args.headed,
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
        )

        context = await browser.new_context(
            accept_downloads=True,

            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),

            viewport={
                "width": 1440,
                "height": 900,
            },
        )

        # ----------------------------------------------------
        # Discover links
        # ----------------------------------------------------

        all_links = []

        page = await context.new_page()

        for source_url in KAESER_DOWNLOADS_URLS:

            try:

                links = await discover_links(
                    page,
                    source_url
                )

                all_links.extend(
                    links
                )

            except Exception as exc:

                logger.warning(
                    "Discovery failed for %s: %s",
                    source_url,
                    exc
                )

        await page.close()

        # Network discovery.
        for source_url in KAESER_DOWNLOADS_URLS:

            try:

                network_links = (
                    await discover_network_requests(
                        browser,
                        source_url
                    )
                )

                all_links.extend(
                    network_links
                )

            except Exception as exc:

                logger.warning(
                    "Network discovery failed: %s",
                    exc
                )

        all_links = deduplicate_links(
            all_links
        )

        save_links(
            all_links
        )

        logger.info(
            "Total unique document links: %d",
            len(all_links)
        )

        # ----------------------------------------------------
        # Discovery only
        # ----------------------------------------------------

        if args.discover_only:

            logger.info(
                "Discovery-only mode. Exiting."
            )

            await browser.close()

            return

        # ----------------------------------------------------
        # Match + download
        # ----------------------------------------------------

        for row in rows:

            document_id = row[
                "document_id"
            ]

            if (
                args.only
                and document_id != args.only
            ):
                continue

            # Already downloaded.
            existing_path = row.get(
                "local_path",
                ""
            )

            if (
                row.get("downloaded") == "true"
                and existing_path
                and Path(existing_path).exists()
            ):

                logger.info(
                    "[SKIP] %s already downloaded",
                    document_id
                )

                continue

            # ----------------------------------------------
            # Existing URL from manifest
            # ----------------------------------------------

            candidate = None

            existing_url = row.get(
                "file_url",
                ""
            ).strip()

            if existing_url:

                candidate = {
                    "href": existing_url,
                    "text": row.get(
                        "title",
                        ""
                    ),
                    "discovery_method": "manifest",
                }

            # ----------------------------------------------
            # Browser discovered match
            # ----------------------------------------------

            if not candidate:

                candidate, top = match_document(
                    row,
                    all_links
                )

                if not candidate:

                    row["download_error"] = (
                        "No confident browser link match"
                    )

                    row["match_score"] = (
                        top[0][0]
                        if top
                        else 0
                    )

                    logger.warning(
                        "[MANUAL] %s | %s | candidates=%s",
                        document_id,
                        row.get("title"),
                        [
                            (
                                score,
                                link.get("href")
                            )
                            for score, link in top
                        ],
                    )

                    continue

            url = candidate[
                "href"
            ]

            score = score_link(
                row,
                candidate
            )

            row["match_score"] = str(
                score
            )

            logger.info(
                "[MATCH] %s -> %s | score=%s",
                document_id,
                url,
                score
            )

            # ----------------------------------------------
            # Download
            # ----------------------------------------------

            destination = (
                OUT_DIR
                /
                f"{document_id}.pdf"
            )

            try:

                await download_pdf(
                    context,
                    url,
                    destination
                )

                size = validate_pdf(
                    destination
                )

                digest = sha256_file(
                    destination
                )

                row.update(
                    {
                        "downloaded": "true",
                        "sha256": digest,
                        "local_path": str(
                            destination
                        ),
                        "retrieved_at": (
                            datetime.datetime.now(
                                datetime.timezone.utc
                            ).isoformat()
                        ),
                        "resolved_file_url": url,
                        "download_error": "",
                    }
                )

                logger.info(
                    "[OK] %s | %.2f MB | SHA256=%s",
                    document_id,
                    size / 1_000_000,
                    digest[:16],
                )

            except Exception as exc:

                row["downloaded"] = "false"

                row["download_error"] = str(
                    exc
                )[:500]

                logger.error(
                    "[FAIL] %s | %s",
                    document_id,
                    exc
                )

                # Delete invalid partial file.
                if destination.exists():

                    try:
                        destination.unlink()
                    except Exception:
                        pass

            await asyncio.sleep(
                args.delay
            )

        await browser.close()

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    save_manifest(
        rows
    )

    logger.info(
        "============================================"
    )

    downloaded = sum(
        row.get("downloaded") == "true"
        for row in rows
    )

    failed = sum(
        bool(row.get("download_error"))
        for row in rows
    )

    logger.info(
        "Downloaded: %d/%d",
        downloaded,
        len(rows)
    )

    logger.info(
        "Failed/unmatched: %d",
        failed
    )

    logger.info(
        "PDF directory: %s",
        OUT_DIR
    )

    logger.info(
        "Updated manifest: %s",
        UPDATED_MANIFEST
    )

    logger.info(
        "Discovered links: %s",
        LINKS_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        logger.info(
            "Interrupted by user."
        )

    except Exception as exc:

        logger.exception(
            "Fatal error: %s",
            exc
        )

        sys.exit(1)