"""
Collect road closure info from PennDOT District 11 press releases.

Queries PennDOT's newsroom via the Coveo search API for press releases about
road closures and restrictions in District 11 (Pittsburgh area). Fetches each
press release page and extracts closure details via regex.

Source: https://www.pa.gov/agencies/penndot/news-and-media/newsroom/district-11
Output: penndot_press_closures_pgh.csv

Requires: pip install requests
"""

import csv
import os
import re
import time
import requests

# Resolve paths relative to project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(DATA_DIR, "penndot_press_closures_pgh.csv")

# Coveo search API for pa.gov (public access token, same as used by the website)
COVEO_ORG = "commonwealthofpennsylvaniaproductiono8jd9ckm"
COVEO_TOKEN = "xx4e57cda9-3464-437d-9375-b947ca6b72c8"
COVEO_URL = f"https://{COVEO_ORG}.org.coveo.com/rest/search/v2"
SEARCH_HUB = "PennDot-News-District-11"

FIELDNAMES = [
    "date", "title", "url",
    "route_number", "road_name", "municipality", "status",
    "from_location", "to_location",
    "data_source",
]

# Regex patterns for extracting closure details from press release text
CLOSURE_PATTERNS = [
    # "Route 28 (East Ohio Street) between X and Y in Z — Closed"
    re.compile(
        r"Route\s+(\d+)\s*\(([^)]+)\)\s+between\s+(.+?)\s+and\s+(.+?)\s+in\s+(.+?)\s*[–—\-]+\s*(Closed|Restricted|Reduced)",
        re.IGNORECASE,
    ),
    # "Route 28 between X and Y in Z — Closed"
    re.compile(
        r"Route\s+(\d+)\s+between\s+(.+?)\s+and\s+(.+?)\s+in\s+(.+?)\s*[–—\-]+\s*(Closed|Restricted|Reduced)",
        re.IGNORECASE,
    ),
    # "SR XXXX (Road Name) ... closed/restricted"
    re.compile(
        r"SR\s+(\d+)\s*\(([^)]+)\).*?(closed|restricted|reduced)",
        re.IGNORECASE,
    ),
    # "Route XXXX (Road Name) ... closed/restricted" (without between)
    re.compile(
        r"Route\s+(\d+)\s*\(([^)]+)\).*?(closed|restricted|reduced)",
        re.IGNORECASE,
    ),
]

MAX_RETRIES = 3
RETRY_BACKOFF = 2
MAX_PAGES_TO_FETCH = 50  # Limit individual page fetches
COVEO_PAGE_SIZE = 100
REQUEST_DELAY = 0.5


def fetch_with_retry(url, params=None, json_body=None, headers=None, method="get"):
    """Fetch with exponential backoff."""
    if headers is None:
        headers = {}
    headers.setdefault("User-Agent", "Mozilla/5.0 (compatible; PittsburghSnowProject/1.0)")

    for attempt in range(MAX_RETRIES):
        try:
            if method == "post":
                resp = requests.post(url, json=json_body, headers=headers, timeout=30)
            else:
                resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = RETRY_BACKOFF ** (attempt + 1)
                print(f"  HTTP {resp.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError:
            wait = RETRY_BACKOFF ** (attempt + 1)
            print(f"  Connection error, retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch after {MAX_RETRIES} retries")


def search_closure_releases():
    """Search PennDOT District 11 newsroom for closure-related press releases via Coveo."""
    all_results = []

    print("Searching PennDOT District 11 press releases via Coveo API...")
    offset = 0
    while True:
        resp = fetch_with_retry(
            COVEO_URL,
            json_body={
                "q": "closure OR closed OR restriction",
                "searchHub": SEARCH_HUB,
                "numberOfResults": COVEO_PAGE_SIZE,
                "firstResult": offset,
            },
            headers={
                "Authorization": f"Bearer {COVEO_TOKEN}",
                "Content-Type": "application/json",
            },
            method="post",
        )

        data = resp.json()
        total = data.get("totalCount", 0)
        results = data.get("results", [])

        if not results:
            break

        all_results.extend(results)
        print(f"  Fetched {len(all_results)}/{total} search results...")

        if len(all_results) >= total:
            break

        offset += COVEO_PAGE_SIZE
        time.sleep(0.3)

    print(f"  Total closure-related press releases found: {len(all_results)}")
    return all_results


def extract_date_from_timestamp(ts):
    """Convert Coveo timestamp (ms since epoch) to date string."""
    if not ts:
        return ""
    try:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError, OSError):
        return ""


def parse_closures_from_text(text, url, date, title):
    """Parse closure details from press release text."""
    rows = []
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean)

    for pattern in CLOSURE_PATTERNS:
        for match in pattern.finditer(clean):
            groups = match.groups()

            if len(groups) == 6:
                rows.append({
                    "date": date, "title": title, "url": url,
                    "route_number": groups[0],
                    "road_name": groups[1].strip(),
                    "from_location": groups[2].strip(),
                    "to_location": groups[3].strip(),
                    "municipality": groups[4].strip(),
                    "status": groups[5].strip().title(),
                    "data_source": "penndot_press",
                })
            elif len(groups) == 5:
                rows.append({
                    "date": date, "title": title, "url": url,
                    "route_number": groups[0],
                    "road_name": "",
                    "from_location": groups[1].strip(),
                    "to_location": groups[2].strip(),
                    "municipality": groups[3].strip(),
                    "status": groups[4].strip().title(),
                    "data_source": "penndot_press",
                })
            elif len(groups) == 3:
                rows.append({
                    "date": date, "title": title, "url": url,
                    "route_number": groups[0],
                    "road_name": groups[1].strip(),
                    "from_location": "", "to_location": "",
                    "municipality": "",
                    "status": groups[2].strip().title(),
                    "data_source": "penndot_press",
                })

    return rows


def main():
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Delete it to re-fetch.")
        return

    # Step 1: Search for closure press releases
    try:
        search_results = search_closure_releases()
    except Exception as e:
        print(f"  Coveo search failed: {e}")
        print("  Writing empty output file.")
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        return

    if not search_results:
        print("  No closure press releases found. Writing empty output.")
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
        return

    # Step 2: Extract closure info from search result snippets (no page fetch needed)
    # Coveo returns title, excerpt, and URI — try parsing from titles first
    all_rows = []
    title_only_count = 0

    for result in search_results:
        title = result.get("title", "")
        uri = result.get("clickUri", "")
        raw = result.get("raw", {})
        date = extract_date_from_timestamp(raw.get("date"))
        excerpt = result.get("excerpt", "")

        # Try to parse from title + excerpt (cheaper than fetching page)
        combined = title + " " + excerpt
        rows = parse_closures_from_text(combined, uri, date, title)
        if rows:
            all_rows.extend(rows)
            title_only_count += len(rows)

    print(f"  Extracted {title_only_count} closures from search result snippets")

    # Step 3: Fetch top press release pages for more detailed extraction
    # Only fetch pages that look most relevant (closure/closed in title)
    closure_results = [
        r for r in search_results
        if re.search(r"clos|restrict", r.get("title", ""), re.IGNORECASE)
    ]
    pages_to_fetch = closure_results[:MAX_PAGES_TO_FETCH]
    print(f"  Fetching {len(pages_to_fetch)} press release pages for detailed parsing...")

    fetched_urls = set()
    for i, result in enumerate(pages_to_fetch):
        uri = result.get("clickUri", "")
        if not uri or uri in fetched_urls:
            continue
        fetched_urls.add(uri)

        raw = result.get("raw", {})
        date = extract_date_from_timestamp(raw.get("date"))
        title = result.get("title", "")

        print(f"  Fetching {i+1}/{len(pages_to_fetch)}: {title[:60]}...")
        try:
            resp = fetch_with_retry(uri)
            rows = parse_closures_from_text(resp.text, uri, date, title)
            # Avoid duplicates from snippet extraction
            for row in rows:
                key = (row["route_number"], row["road_name"], row["status"], row["url"])
                if not any(
                    (r["route_number"], r["road_name"], r["status"], r["url"]) == key
                    for r in all_rows
                ):
                    all_rows.append(row)
        except Exception as e:
            print(f"    Error: {e}")

        time.sleep(REQUEST_DELAY)

    # Write output
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} closure records to {OUTPUT_CSV}")

    if all_rows:
        statuses = {}
        for r in all_rows:
            s = r["status"] or "(unknown)"
            statuses[s] = statuses.get(s, 0) + 1
        print("  By status:")
        for s, count in sorted(statuses.items(), key=lambda x: -x[1]):
            print(f"    {s:25s} : {count}")


if __name__ == "__main__":
    main()
