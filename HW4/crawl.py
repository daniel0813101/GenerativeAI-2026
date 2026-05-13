"""Crawler for NYCU HW4 PTT Stock crawl task."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
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
REQUEST_MAX_RETRIES = 5
REQUEST_RETRY_BACKOFF_SECONDS = 1.0
SAMPLE_EVERY_N_PAGES = 25
IMAGE_URL_PATTERN = re.compile(
    r"https?://[^\s\"'<>]+?\.(?:jpg|jpeg|png|gif)(?=$|[\s\"'<>])",
    re.IGNORECASE,
)
MONTH_NAMES = {
    "01": "January",
    "02": "February",
    "03": "March",
    "04": "April",
    "05": "May",
    "06": "June",
    "07": "July",
    "08": "August",
    "09": "September",
    "10": "October",
    "11": "November",
    "12": "December",
}

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


@dataclass
class CrawlStats:
    """Mutable counters and timing data for one crawl run."""

    started_at: float
    pages: int = 0
    articles: int = 0
    popular: int = 0
    phase: str = "seeking-2025-12-31"
    current_date: str | None = None
    current_month: str | None = None
    month_articles: int = 0
    month_popular: int = 0
    last_status_length: int = 0
    status_updates: int = 0
    status_started: bool = False


@dataclass(frozen=True)
class ArticleRecord:
    """Article data loaded from ``articles.jsonl``."""

    date: str
    title: str
    url: str


def fetch(url: str) -> str:
    """Fetches a URL and returns decoded HTML.

    Args:
        url: The absolute URL to fetch.

    Returns:
        The response body decoded as text.

    Raises:
        requests.HTTPError: If the server returns an unsuccessful status code.
        requests.RequestException: If the request fails after all retries.
    """
    last_error: requests.RequestException | None = None
    for attempt in range(1, REQUEST_MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_SLEEP_SECONDS)
            response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            response.encoding = "utf-8"
            return response.text
        except requests.RequestException as error:
            last_error = error
            if attempt == REQUEST_MAX_RETRIES:
                break
            sleep_seconds = REQUEST_RETRY_BACKOFF_SECONDS * attempt
            print(
                "[retry] "
                f"attempt={attempt}/{REQUEST_MAX_RETRIES} "
                f"sleep={sleep_seconds:.1f}s url={url} error={error}",
                flush=True,
            )
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error


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


def load_articles(path: str) -> list[ArticleRecord]:
    """Loads article records from a JSON Lines file.

    Args:
        path: Path to ``articles.jsonl`` or another compatible JSONL file.

    Returns:
        Article records loaded from the file.
    """
    articles: list[ArticleRecord] = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            articles.append(
                ArticleRecord(
                    date=str(item["date"]),
                    title=str(item["title"]),
                    url=str(item["url"]),
                )
            )
    return articles


def filter_articles_by_date(
    articles: Iterable[ArticleRecord],
    start_date: str,
    end_date: str,
) -> list[ArticleRecord]:
    """Filters articles by inclusive ``MMDD`` date range.

    Args:
        articles: Article records to filter.
        start_date: Inclusive start date in ``MMDD`` format.
        end_date: Inclusive end date in ``MMDD`` format.

    Returns:
        Articles whose date falls within the requested range.
    """
    return [
        article
        for article in articles
        if start_date <= article.date <= end_date
    ]


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


def format_elapsed(seconds: float) -> str:
    """Formats elapsed seconds as a compact duration string.

    Args:
        seconds: Number of elapsed seconds.

    Returns:
        A ``HH:MM:SS`` duration string.
    """
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_status_message(stats: CrawlStats) -> str:
    """Builds the one-line crawl status message.

    Args:
        stats: Current crawl counters and timing data.

    Returns:
        A single-line status message.
    """
    elapsed = format_elapsed(time.monotonic() - stats.started_at)
    return (
        "[crawl] "
        f"update={stats.status_updates} phase={stats.phase} "
        f"pages={stats.pages} articles={stats.articles} "
        f"popular={stats.popular} date={stats.current_date or '-'} "
        f"elapsed={elapsed}"
    )


def clear_status_line(stats: CrawlStats) -> None:
    """Clears the currently displayed terminal status line.

    Args:
        stats: Current crawl counters and terminal status metadata.
    """
    if stats.last_status_length and sys.stderr.isatty():
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()
        stats.last_status_length = 0


def update_status_line(stats: CrawlStats) -> None:
    """Refreshes the crawl status line.

    Args:
        stats: Current crawl counters and timing data.
    """
    if not sys.stderr.isatty():
        return

    stats.status_updates += 1
    message = build_status_message(stats)
    sys.stderr.write("\r\033[K" + message)
    sys.stderr.flush()
    stats.last_status_length = len(message)
    stats.status_started = True


def print_crawl_event(stats: CrawlStats, message: str) -> None:
    """Prints a crawl event without leaving a stale status line behind.

    Args:
        stats: Current crawl counters and terminal status metadata.
        message: Event message to print.
    """
    clear_status_line(stats)
    print(message, flush=True)
    update_status_line(stats)


def print_month_finish(stats: CrawlStats, finished_month: str) -> None:
    """Prints a message when crawling finishes a month.

    Args:
        stats: Current crawl counters and terminal status metadata.
        finished_month: The two-digit month that has just been completed.
    """
    month_name = MONTH_NAMES.get(finished_month, finished_month)
    print_crawl_event(
        stats,
        (
            f"{month_name} Finish: "
            f"articles={stats.month_articles}, popular={stats.month_popular}"
        ),
    )
    stats.month_articles = 0
    stats.month_popular = 0


def print_crawl_summary(stats: CrawlStats, articles_path: str, popular_path: str) -> None:
    """Prints the final crawl summary.

    Args:
        stats: Final crawl counters and timing data.
        articles_path: Path to the generated ``articles.jsonl`` file.
        popular_path: Path to the generated ``popular_articles.jsonl`` file.
    """
    clear_status_line(stats)
    elapsed = format_elapsed(time.monotonic() - stats.started_at)
    print("Crawl summary", flush=True)
    print(f"- pages scanned: {stats.pages}", flush=True)
    print(f"- articles written: {stats.articles}", flush=True)
    print(f"- popular articles written: {stats.popular}", flush=True)
    print(f"- articles file: {articles_path}", flush=True)
    print(f"- popular articles file: {popular_path}", flush=True)
    print(f"- elapsed time: {elapsed}", flush=True)


def extract_push_counts(soup: BeautifulSoup) -> tuple[Counter[str], Counter[str]]:
    """Extracts push and boo user counters from one article page.

    Args:
        soup: A BeautifulSoup document for an article page.

    Returns:
        A tuple ``(push_counter, boo_counter)`` keyed by user ID.
    """
    push_counter: Counter[str] = Counter()
    boo_counter: Counter[str] = Counter()

    for push_node in soup.select("div.push"):
        tag_node = push_node.select_one("span.push-tag")
        user_node = push_node.select_one("span.push-userid")
        if tag_node is None or user_node is None:
            continue

        tag = tag_node.get_text(strip=True)
        user_id = user_node.get_text()
        if tag == "推":
            push_counter[user_id] += 1
        elif tag == "噓":
            boo_counter[user_id] += 1

    return push_counter, boo_counter


def merge_counters(target: Counter[str], source: Counter[str]) -> None:
    """Adds counts from one counter into another counter.

    Args:
        target: Counter to mutate.
        source: Counter whose counts should be added.
    """
    target.update(source)


def format_top10(counter: Counter[str]) -> list[dict[str, int | str]]:
    """Formats the top 10 entries from a user counter.

    Args:
        counter: Counter keyed by user ID.

    Returns:
        A list of ``{"user_id": user_id, "count": count}`` dictionaries.
    """
    return [
        {"user_id": user_id, "count": count}
        for user_id, count in counter.most_common(10)
    ]


def write_json(path: str, item: dict) -> None:
    """Writes a JSON object to disk.

    Args:
        path: Output JSON path.
        item: JSON-serializable object to write.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(item, file, ensure_ascii=False, indent=2)


def extract_image_urls(soup: BeautifulSoup) -> list[str]:
    """Extracts image URLs from an article page.

    Args:
        soup: A BeautifulSoup document for an article page.

    Returns:
        Image URLs matching the assignment definition. Duplicates are retained.
    """
    text = soup.get_text("\n")
    return IMAGE_URL_PATTERN.findall(text)


def print_push_status(processed: int, total: int, article: ArticleRecord) -> None:
    """Prints a compact push-processing progress line.

    Args:
        processed: Number of processed articles.
        total: Total number of articles to process.
        article: The current article being processed.
    """
    if not sys.stderr.isatty():
        return
    message = (
        f"[push] {processed}/{total} date={article.date} "
        f"url={article.url}"
    )
    sys.stderr.write("\r\033[K" + message)
    sys.stderr.flush()


def print_push_summary(
    output_path: str,
    article_count: int,
    push_total: int,
    boo_total: int,
    started_at: float,
) -> None:
    """Prints the final push summary.

    Args:
        output_path: Path to the generated push JSON file.
        article_count: Number of articles processed.
        push_total: Total push comments counted.
        boo_total: Total boo comments counted.
        started_at: Monotonic start time of the push command.
    """
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()
    elapsed = format_elapsed(time.monotonic() - started_at)
    print("Push summary", flush=True)
    print(f"- articles processed: {article_count}", flush=True)
    print(f"- push total: {push_total}", flush=True)
    print(f"- boo total: {boo_total}", flush=True)
    print(f"- output file: {output_path}", flush=True)
    print(f"- elapsed time: {elapsed}", flush=True)


def print_popular_status(processed: int, total: int, article: ArticleRecord) -> None:
    """Prints a compact popular-processing progress line.

    Args:
        processed: Number of processed articles.
        total: Total number of articles to process.
        article: The current article being processed.
    """
    if not sys.stderr.isatty():
        return
    message = (
        f"[popular] {processed}/{total} date={article.date} "
        f"url={article.url}"
    )
    sys.stderr.write("\r\033[K" + message)
    sys.stderr.flush()


def print_popular_summary(
    output_path: str,
    article_count: int,
    image_count: int,
    started_at: float,
) -> None:
    """Prints the final popular summary.

    Args:
        output_path: Path to the generated popular JSON file.
        article_count: Number of popular articles processed.
        image_count: Number of extracted image URLs.
        started_at: Monotonic start time of the popular command.
    """
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()
    elapsed = format_elapsed(time.monotonic() - started_at)
    print("Popular summary", flush=True)
    print(f"- popular articles processed: {article_count}", flush=True)
    print(f"- image URLs extracted: {image_count}", flush=True)
    print(f"- output file: {output_path}", flush=True)
    print(f"- elapsed time: {elapsed}", flush=True)


def push(start_date: str, end_date: str, output_dir: str = ".") -> None:
    """Computes push and boo statistics for a date range.

    Args:
        start_date: Inclusive start date in ``MMDD`` format.
        end_date: Inclusive end date in ``MMDD`` format.
        output_dir: Directory containing ``articles.jsonl`` and receiving the
            generated ``push_{start_date}_{end_date}.json`` file.
    """
    started_at = time.monotonic()
    articles_path = os.path.join(output_dir, "articles.jsonl")
    output_path = os.path.join(output_dir, f"push_{start_date}_{end_date}.json")
    articles = filter_articles_by_date(
        load_articles(articles_path),
        start_date,
        end_date,
    )

    push_counter: Counter[str] = Counter()
    boo_counter: Counter[str] = Counter()
    total = len(articles)

    for index, article in enumerate(articles, start=1):
        print_push_status(index, total, article)
        article_push_counter, article_boo_counter = extract_push_counts(
            fetch_soup(article.url)
        )
        merge_counters(push_counter, article_push_counter)
        merge_counters(boo_counter, article_boo_counter)

    output = {
        "push": {
            "total": sum(push_counter.values()),
            "top10": format_top10(push_counter),
        },
        "boo": {
            "total": sum(boo_counter.values()),
            "top10": format_top10(boo_counter),
        },
    }
    write_json(output_path, output)
    print_push_summary(
        output_path,
        total,
        output["push"]["total"],
        output["boo"]["total"],
        started_at,
    )


def popular(start_date: str, end_date: str, output_dir: str = ".") -> None:
    """Extracts image URLs from popular articles in a date range.

    Args:
        start_date: Inclusive start date in ``MMDD`` format.
        end_date: Inclusive end date in ``MMDD`` format.
        output_dir: Directory containing ``popular_articles.jsonl`` and
            receiving the generated ``popular_{start_date}_{end_date}.json``.
    """
    started_at = time.monotonic()
    articles_path = os.path.join(output_dir, "popular_articles.jsonl")
    output_path = os.path.join(output_dir, f"popular_{start_date}_{end_date}.json")
    articles = filter_articles_by_date(
        load_articles(articles_path),
        start_date,
        end_date,
    )

    image_urls: list[str] = []
    total = len(articles)

    for index, article in enumerate(articles, start=1):
        print_popular_status(index, total, article)
        image_urls.extend(extract_image_urls(fetch_soup(article.url)))

    output = {
        "number_of_popular_articles": total,
        "image_urls": image_urls,
    }
    write_json(output_path, output)
    print_popular_summary(output_path, total, len(image_urls), started_at)


def extract_keyword_search_text(soup: BeautifulSoup) -> str | None:
    """Extracts the article text range used for keyword matching.

    The assignment defines the searchable range as starting from ``作者``
    inclusive and ending before ``※ 發信站``. Articles without ``※ 發信站`` are
    ignored.

    Args:
        soup: A BeautifulSoup document for an article page.

    Returns:
        The searchable article text, or ``None`` if the required range cannot
        be found.
    """
    main_content = soup.select_one("#main-content")
    text = main_content.get_text("\n") if main_content else soup.get_text("\n")
    start = text.find("作者")
    end = text.find("※ 發信站")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end]


def article_matches_keyword(soup: BeautifulSoup, keyword: str) -> bool:
    """Checks whether an article page matches a keyword.

    Args:
        soup: A BeautifulSoup document for an article page.
        keyword: Keyword to search for.

    Returns:
        ``True`` when the keyword appears in the assignment-defined article
        body range, otherwise ``False``.
    """
    search_text = extract_keyword_search_text(soup)
    return search_text is not None and keyword in search_text


def print_keyword_status(processed: int, total: int, matched: int, article: ArticleRecord) -> None:
    """Prints a compact keyword-processing progress line.

    Args:
        processed: Number of processed articles.
        total: Total number of articles to process.
        matched: Number of keyword-matched articles so far.
        article: The current article being processed.
    """
    if not sys.stderr.isatty():
        return
    message = (
        f"[keyword] {processed}/{total} matched={matched} "
        f"date={article.date} url={article.url}"
    )
    sys.stderr.write("\r\033[K" + message)
    sys.stderr.flush()


def print_keyword_summary(
    output_path: str,
    article_count: int,
    matched_count: int,
    image_count: int,
    started_at: float,
) -> None:
    """Prints the final keyword summary.

    Args:
        output_path: Path to the generated keyword JSON file.
        article_count: Number of articles processed.
        matched_count: Number of articles matching the keyword.
        image_count: Number of extracted image URLs.
        started_at: Monotonic start time of the keyword command.
    """
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()
    elapsed = format_elapsed(time.monotonic() - started_at)
    print("Keyword summary", flush=True)
    print(f"- articles processed: {article_count}", flush=True)
    print(f"- matched articles: {matched_count}", flush=True)
    print(f"- image URLs extracted: {image_count}", flush=True)
    print(f"- output file: {output_path}", flush=True)
    print(f"- elapsed time: {elapsed}", flush=True)


def keyword(start_date: str, end_date: str, search_keyword: str, output_dir: str = ".") -> None:
    """Extracts image URLs from articles whose body contains a keyword.

    Args:
        start_date: Inclusive start date in ``MMDD`` format.
        end_date: Inclusive end date in ``MMDD`` format.
        search_keyword: Keyword to match inside the assignment-defined article
            body range.
        output_dir: Directory containing ``articles.jsonl`` and receiving the
            generated keyword output JSON file.
    """
    started_at = time.monotonic()
    articles_path = os.path.join(output_dir, "articles.jsonl")
    output_path = os.path.join(
        output_dir,
        f"keyword_{start_date}_{end_date}_{search_keyword}.json",
    )
    articles = filter_articles_by_date(
        load_articles(articles_path),
        start_date,
        end_date,
    )

    image_urls: list[str] = []
    matched_count = 0
    total = len(articles)

    for index, article in enumerate(articles, start=1):
        print_keyword_status(index, total, matched_count, article)
        soup = fetch_soup(article.url)
        if not article_matches_keyword(soup, search_keyword):
            continue
        matched_count += 1
        image_urls.extend(extract_image_urls(soup))
        print_keyword_status(index, total, matched_count, article)

    output = {"image_urls": image_urls}
    write_json(output_path, output)
    print_keyword_summary(
        output_path,
        total,
        matched_count,
        len(image_urls),
        started_at,
    )


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
    stats = CrawlStats(started_at=time.monotonic())
    update_status_line(stats)

    try:
        while url:
            stats.pages += 1
            soup = fetch_soup(url)
            entries = list(reversed(parse_index_entries(soup)))
            previous_url = parse_previous_page_url(soup)
            stats.current_date = entries[0].date if entries else None
            if not collecting:
                update_status_line(stats)

            if not collecting:
                year = first_year_for_date(entries, "1231")
                if year == TARGET_YEAR:
                    collecting = True
                    stats.phase = "collecting-2025"
                    print_crawl_event(stats, "Start collecting: confirmed 2025-12-31")
                else:
                    url = previous_url
                    continue

            if stats.pages % SAMPLE_EVERY_N_PAGES == 0:
                year = sample_year(entries)
                if year is not None and year < TARGET_YEAR:
                    stats.phase = "stopped-before-2025"
                    update_status_line(stats)
                    break

            for entry in entries:
                stats.current_date = entry.date
                collect, stop = should_collect_entry(entry, seen_jan_first)
                if collect:
                    month = entry.date[:2]
                    if stats.current_month is None:
                        stats.current_month = month
                    elif month != stats.current_month:
                        print_month_finish(stats, stats.current_month)
                        stats.current_month = month

                    write_article_outputs(entry, articles_path, popular_path)
                    stats.articles += 1
                    stats.month_articles += 1
                    if entry.push_mark == "爆":
                        stats.popular += 1
                        stats.month_popular += 1
                    if entry.date == "0101":
                        seen_jan_first = True
                    update_status_line(stats)

                if stop:
                    stats.phase = "done"
                    update_status_line(stats)
                    url = None
                    break

            if stats.phase == "done":
                break
            url = previous_url
    finally:
        if stats.current_month is not None and stats.phase == "done":
            print_month_finish(stats, stats.current_month)
        print_crawl_summary(stats, articles_path, popular_path)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        The parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="PTT Stock HW4 crawler.")
    subparsers = parser.add_subparsers(dest="command")

    crawl_parser = subparsers.add_parser(
        "crawl",
        help="Crawl 2025 PTT Stock articles.",
    )
    crawl_parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for articles.jsonl and popular_articles.jsonl.",
    )

    push_parser = subparsers.add_parser(
        "push",
        help="Compute push and boo statistics for a date range.",
    )
    push_parser.add_argument("start_date", help="Inclusive start date in MMDD format.")
    push_parser.add_argument("end_date", help="Inclusive end date in MMDD format.")
    push_parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory containing articles.jsonl and receiving push output.",
    )

    popular_parser = subparsers.add_parser(
        "popular",
        help="Extract image URLs from popular articles in a date range.",
    )
    popular_parser.add_argument("start_date", help="Inclusive start date in MMDD format.")
    popular_parser.add_argument("end_date", help="Inclusive end date in MMDD format.")
    popular_parser.add_argument(
        "--output-dir",
        default=".",
        help=(
            "Directory containing popular_articles.jsonl and receiving "
            "popular output."
        ),
    )

    keyword_parser = subparsers.add_parser(
        "keyword",
        help="Extract image URLs from articles whose body contains a keyword.",
    )
    keyword_parser.add_argument("start_date", help="Inclusive start date in MMDD format.")
    keyword_parser.add_argument("end_date", help="Inclusive end date in MMDD format.")
    keyword_parser.add_argument("keyword", help="Keyword without whitespace.")
    keyword_parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory containing articles.jsonl and receiving keyword output.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the command-line entry point."""
    args = parse_args()
    if args.command in (None, "crawl"):
        crawl(args.output_dir)
    elif args.command == "push":
        push(args.start_date, args.end_date, args.output_dir)
    elif args.command == "popular":
        popular(args.start_date, args.end_date, args.output_dir)
    elif args.command == "keyword":
        keyword(args.start_date, args.end_date, args.keyword, args.output_dir)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
