"""Crawler for NYCU HW4 PTT Stock crawl task."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.ptt.cc"
STOCK_INDEX_URL = f"{BASE_URL}/bbs/Stock/index.html"
TARGET_YEAR = 2025
REQUEST_TIMEOUT = 15
REQUEST_SLEEP_SECONDS = 0.05
SAMPLE_EVERY_N_PAGES = 25

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
)


@dataclass(frozen=True)
class IndexEntry:
    """Article data parsed from a PTT list page."""

    date: str
    title: str
    url: str
    push_mark: str


def fetch(url: str) -> str:
    """Fetches a URL and returns decoded HTML.

    Args:
        url: The absolute URL to fetch.

    Returns:
        The response body decoded as text.

    Raises:
        requests.HTTPError: If the server returns an unsuccessful status code.
        requests.RequestException: If the request fails.
    """
    time.sleep(REQUEST_SLEEP_SECONDS)
    response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def fetch_soup(url: str) -> BeautifulSoup:
    """Fetches a URL and parses the response with BeautifulSoup.

    Args:
        url: The absolute URL to fetch.

    Returns:
        A BeautifulSoup document parsed from the fetched HTML.
    """
    return BeautifulSoup(fetch(url), "html.parser")


def normalize_url(href: str) -> str:
    """Converts a PTT href into an absolute HTTPS URL.

    Args:
        href: A relative or absolute URL from a PTT page.

    Returns:
        An absolute URL beginning with ``https://``.
    """
    url = urljoin(BASE_URL, href)
    return re.sub(r"^http://", "https://", url)


def normalize_list_date(raw_date: str) -> str | None:
    """Converts a list-page date like ``5/11`` into ``MMDD``.

    Args:
        raw_date: The date text shown on a PTT board list page.

    Returns:
        A four-digit ``MMDD`` string, or ``None`` if parsing fails.
    """
    match = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})\s*", raw_date)
    if not match:
        return None
    month = int(match.group(1))
    day = int(match.group(2))
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return None
    return f"{month:02d}{day:02d}"


def should_skip_title(title: str) -> bool:
    """Checks whether an article title should be ignored by the spec.

    Args:
        title: The title text from the PTT list page.

    Returns:
        ``True`` if the article should be skipped, otherwise ``False``.
    """
    clean_title = title.strip()
    return (
        not clean_title
        or "[公告]" in clean_title
        or "Fw:[公告]" in clean_title
    )


def parse_index_entries(soup: BeautifulSoup) -> list[IndexEntry]:
    """Parses article rows from a PTT Stock list page.

    Args:
        soup: A BeautifulSoup document for a board index page.

    Returns:
        A list of valid-looking index entries. The order matches the page order.
    """
    entries: list[IndexEntry] = []
    for row in soup.select("div.r-ent"):
        title_node = row.select_one("div.title a")
        date_node = row.select_one("div.date")
        push_node = row.select_one("div.nrec")
        if title_node is None or date_node is None:
            continue

        title = title_node.get_text(strip=True)
        href = title_node.get("href")
        date = normalize_list_date(date_node.get_text())
        if not href or date is None or should_skip_title(title):
            continue

        entries.append(
            IndexEntry(
                date=date,
                title=title,
                url=normalize_url(href),
                push_mark=push_node.get_text(strip=True) if push_node else "",
            )
        )
    return entries


def parse_previous_page_url(soup: BeautifulSoup) -> str | None:
    """Finds the previous-page URL on a PTT board list page.

    Args:
        soup: A BeautifulSoup document for a board index page.

    Returns:
        The absolute previous-page URL, or ``None`` if no link is found.
    """
    for link in soup.select("div.btn-group-paging a.btn"):
        if "上頁" in link.get_text():
            href = link.get("href")
            return normalize_url(href) if href else None
    return None


def parse_article_year(soup: BeautifulSoup) -> int | None:
    """Parses the year from an article's internal PTT time metadata.

    Args:
        soup: A BeautifulSoup document for an article page.

    Returns:
        The four-digit article year, or ``None`` if the time metadata is absent
        or cannot be parsed.
    """
    tags = soup.select("span.article-meta-tag")
    values = soup.select("span.article-meta-value")
    for tag, value in zip(tags, values):
        if tag.get_text(strip=True) != "時間":
            continue
        raw_time = value.get_text(" ", strip=True)
        try:
            return datetime.strptime(raw_time, "%a %b %d %H:%M:%S %Y").year
        except ValueError:
            match = re.search(r"\b(20\d{2})\b", raw_time)
            return int(match.group(1)) if match else None
    return None


def get_article_year(url: str) -> int | None:
    """Fetches an article and returns its internal metadata year.

    Args:
        url: The absolute article URL.

    Returns:
        The four-digit article year, or ``None`` if it cannot be determined.
    """
    try:
        return parse_article_year(fetch_soup(url))
    except requests.RequestException:
        return None


def first_year_for_date(entries: Iterable[IndexEntry], date: str) -> int | None:
    """Finds the first parseable article year for a list-page date.

    Args:
        entries: Candidate index entries from one page.
        date: The ``MMDD`` date to sample, such as ``1231``.

    Returns:
        The first successfully parsed article year for that date, or ``None``.
    """
    for entry in entries:
        if entry.date != date:
            continue
        year = get_article_year(entry.url)
        if year is not None:
            return year
    return None


def sample_year(entries: Iterable[IndexEntry]) -> int | None:
    """Samples one article from a page and returns its internal year.

    Args:
        entries: Candidate index entries from one page.

    Returns:
        The parsed year from the first sample that succeeds, or ``None``.
    """
    for entry in entries:
        year = get_article_year(entry.url)
        if year is not None:
            return year
    return None


def append_jsonl(path: str, item: dict[str, str]) -> None:
    """Appends one JSON object to a JSON Lines file.

    Args:
        path: The output JSONL path.
        item: The JSON-serializable article object to write.
    """
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_article_outputs(entry: IndexEntry, articles_path: str, popular_path: str) -> None:
    """Writes an article to crawl outputs, including popular output if needed.

    Args:
        entry: The list-page article entry to write.
        articles_path: Path to ``articles.jsonl``.
        popular_path: Path to ``popular_articles.jsonl``.
    """
    item = {"date": entry.date, "title": entry.title, "url": entry.url}
    append_jsonl(articles_path, item)
    if entry.push_mark == "爆":
        append_jsonl(popular_path, item)


def remove_existing_outputs(paths: Iterable[str]) -> None:
    """Removes stale output files before starting a fresh crawl.

    Args:
        paths: Output paths to remove if they exist.
    """
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def should_collect_entry(entry: IndexEntry, seen_jan_first: bool) -> tuple[bool, bool]:
    """Decides whether to collect an entry after crawl has entered 2025.

    Boundary dates are verified using article-page metadata because list-page
    dates do not include the year.

    Args:
        entry: The candidate article entry.
        seen_jan_first: Whether a 2025 ``01/01`` article has already appeared.

    Returns:
        A tuple ``(collect, stop)``. ``collect`` means write this article, and
        ``stop`` means the crawler has crossed into articles older than 2025.
    """
    if entry.date == "0101":
        year = get_article_year(entry.url)
        return year == TARGET_YEAR, year is not None and year < TARGET_YEAR

    if seen_jan_first and entry.date == "1231":
        year = get_article_year(entry.url)
        if year is None:
            return False, False
        return year == TARGET_YEAR, year < TARGET_YEAR

    return True, False


def crawl(output_dir: str = ".") -> None:
    """Crawls PTT Stock 2025 articles and writes HW4 crawl outputs.

    The crawler walks board pages backward from the latest page. It starts
    collecting only after confirming that a list-page ``12/31`` article belongs
    to 2025, then stops after confirming it has crossed before ``01/01`` 2025.

    Args:
        output_dir: Directory where ``articles.jsonl`` and
            ``popular_articles.jsonl`` will be written.
    """
    os.makedirs(output_dir, exist_ok=True)
    articles_path = os.path.join(output_dir, "articles.jsonl")
    popular_path = os.path.join(output_dir, "popular_articles.jsonl")
    remove_existing_outputs([articles_path, popular_path])

    url: str | None = STOCK_INDEX_URL
    collecting = False
    seen_jan_first = False
    page_count = 0

    while url:
        page_count += 1
        soup = fetch_soup(url)
        entries = list(reversed(parse_index_entries(soup)))
        previous_url = parse_previous_page_url(soup)

        if not collecting:
            year = first_year_for_date(entries, "1231")
            if year == TARGET_YEAR:
                collecting = True
            else:
                url = previous_url
                continue

        if page_count % SAMPLE_EVERY_N_PAGES == 0:
            year = sample_year(entries)
            if year is not None and year < TARGET_YEAR:
                break

        for entry in entries:
            collect, stop = should_collect_entry(entry, seen_jan_first)
            if collect:
                write_article_outputs(entry, articles_path, popular_path)
                if entry.date == "0101":
                    seen_jan_first = True
            if stop:
                return

        url = previous_url


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        The parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Crawl PTT Stock 2025 articles.")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for articles.jsonl and popular_articles.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the crawl command-line entry point."""
    args = parse_args()
    crawl(args.output_dir)


if __name__ == "__main__":
    main()
