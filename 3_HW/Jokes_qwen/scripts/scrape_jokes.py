import argparse
import random
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://anekdot.ru/random/anekdot/"
LAST_URL_TEMPLATE = "https://www.anekdot.ru/last/anekdot/?page={page}"


def build_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False  # игнорируем системные прокси, чтобы не уводило на 127.0.0.1
    retries = Retry(total=3, backoff_factor=0.6, status_forcelist=(500, 502, 503, 504))
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "anekdot-scraper/0.2"})
    session.verify = False
    session.proxies = {"http": None, "https": None}
    return session


def fetch_random_page(session: requests.Session) -> list[str]:
    resp = session.get(BASE_URL, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    jokes = []
    for div in soup.select("div.text"):
        text = div.get_text(separator=" ", strip=True)
        text = text.replace("\u200b", "").strip()
        if text:
            jokes.append(text)
    return jokes


def scrape_random(pages: int, pause: tuple[float, float]) -> list[str]:
    session = build_session()
    all_jokes: list[str] = []
    seen: set[str] = set()
    low, high = pause
    for idx in range(pages):
        jokes = fetch_random_page(session)
        for j in jokes:
            if j not in seen:
                seen.add(j)
                all_jokes.append(j)
        if (idx + 1) % 5 == 0:
            print(f"[scrape] page {idx + 1}/{pages}, collected {len(all_jokes)} jokes")
        sleep_for = random.uniform(low, high)
        time.sleep(sleep_for)
    return all_jokes


def scrape_last(pages: int, start_page: int, pause: tuple[float, float]) -> list[str]:
    session = build_session()
    all_jokes: list[str] = []
    seen: set[str] = set()
    low, high = pause
    for page in range(start_page, start_page + pages):
        url = LAST_URL_TEMPLATE.format(page=page)
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        jokes = []
        for div in soup.select("div.text"):
            text = div.get_text(separator=" ", strip=True)
            text = text.replace("\u200b", "").strip()
            if text:
                jokes.append(text)
        for j in jokes:
            if j not in seen:
                seen.add(j)
                all_jokes.append(j)
        if (page - start_page + 1) % 10 == 0:
            print(f"[scrape:last] page {page}/{start_page + pages - 1}, collected {len(all_jokes)} jokes")
        time.sleep(random.uniform(low, high))
    return all_jokes


def main():
    parser = argparse.ArgumentParser(description="Scrape Russian jokes from anekdot.ru")
    parser.add_argument("--source", choices=["random", "last"], default="random", help="Use random feed or paginated /last/")
    parser.add_argument("--pages", type=int, default=120, help="How many pages to pull")
    parser.add_argument("--start_page", type=int, default=1, help="Start page for /last/ mode")
    parser.add_argument("--out", type=Path, default=Path("data/jokes_raw.txt"), help="Where to write jokes")
    parser.add_argument("--pause", type=float, nargs=2, default=(0.5, 1.2), metavar=("MIN", "MAX"), help="Random sleep bounds between requests")
    args = parser.parse_args()

    pause = tuple(args.pause)
    if args.source == "random":
        jokes = scrape_random(args.pages, pause)
    else:
        jokes = scrape_last(args.pages, args.start_page, pause)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for j in jokes:
            f.write(j.replace("\r", " ").replace("\n", " ").strip() + "\n")
    print(f"[done] wrote {len(jokes)} jokes to {args.out}")


if __name__ == "__main__":
    main()
