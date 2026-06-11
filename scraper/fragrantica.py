"""
Scrapes fragrance data from Fragrantica — name, brand, notes, accords, concentration.
Run directly: python -m scraper.fragrantica
"""

import csv
import logging
import random
import time
from pathlib import Path

from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://www.fragrantica.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# (brand_slug_on_fragrantica, price_tier)
BRANDS = [
    # Luxury — typically $150+
    ("Creed", "luxury"),
    ("Tom-Ford", "luxury"),
    ("Maison-Margiela", "luxury"),
    ("Parfums-de-Marly", "luxury"),
    ("Xerjoff", "luxury"),
    ("Amouage", "luxury"),
    ("Frederic-Malle", "luxury"),
    ("By-Kilian", "luxury"),
    ("Initio-Parfums-Prives", "luxury"),
    ("Clive-Christian", "luxury"),
    ("Serge-Lutens", "luxury"),
    ("Diptyque", "luxury"),
    ("Jo-Malone-London", "luxury"),
    ("Chanel", "luxury"),
    ("Dior", "luxury"),
    ("Guerlain", "luxury"),
    ("Hermes", "luxury"),
    ("Yves-Saint-Laurent", "luxury"),
    ("Giorgio-Armani", "luxury"),
    ("Givenchy", "luxury"),
    # Affordable — typically under $60
    ("Armaf", "affordable"),
    ("Lattafa-Perfumes", "affordable"),
    ("Al-Haramain-Perfumes", "affordable"),
    ("Rasasi", "affordable"),
    ("Fragrance-World", "affordable"),
    ("Maison-Alhambra", "affordable"),
    ("Afnan-Perfumes", "affordable"),
    ("Al-Rehab", "affordable"),
    ("Ajmal", "affordable"),
    ("Wadi-al-Khaleej", "affordable"),
    ("Dua-Fragrances", "affordable"),
    ("Alexandre-J", "affordable"),
    ("Zara", "affordable"),
    ("Paco-Rabanne", "affordable"),
    ("Calvin-Klein", "affordable"),
    ("Hugo-Boss", "affordable"),
    ("Davidoff", "affordable"),
    ("Azzaro", "affordable"),
    ("Nautica", "affordable"),
    ("Adidas", "affordable"),
]

FIELDS = [
    "name", "brand", "price_tier", "url",
    "concentration", "top_notes", "middle_notes", "base_notes", "main_accords",
]


def _get(url: str, session: requests.Session) -> Optional[BeautifulSoup]:
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except Exception as exc:
        log.warning("Failed %s: %s", url, exc)
        return None


def _parse_brand_page(soup: BeautifulSoup) -> list[str]:
    """Return relative fragrance URLs listed on a brand index page."""
    links = []
    for a in soup.select("a[href*='/perfume/']"):
        href = a.get("href", "")
        # Brand pages list frags as /perfume/Brand/Name-ID.html
        if href.startswith("/perfume/") and href.endswith(".html"):
            links.append(href)
    return list(dict.fromkeys(links))  # dedup while preserving order


def _parse_notes_from_section(section) -> list[str]:
    """Extract note names from a pyramid tier section (images or text spans)."""
    notes = []
    # Primary: img alt text
    for img in section.select("img[alt]"):
        alt = img["alt"].strip()
        if alt and alt.lower() not in ("", "note"):
            notes.append(alt)
    # Fallback: h3.caption text (Fragrantica sometimes renders both)
    if not notes:
        for caption in section.select("h3.caption, .caption"):
            txt = caption.get_text(strip=True)
            if txt:
                notes.append(txt)
    return notes


def _parse_fragrance_page(soup: BeautifulSoup, url: str, brand: str, tier: str) -> Optional[dict]:
    """Parse a single Fragrantica fragrance page into a flat dict."""
    # Name
    name_el = soup.find("h1", itemprop="name") or soup.find("h1")
    if not name_el:
        log.debug("No <h1> found on %s", url)
        return None
    name = name_el.get_text(strip=True)

    # Concentration — look for common keywords near the name
    concentration = ""
    for span in soup.find_all(["span", "p", "div"]):
        txt = span.get_text(strip=True)
        for kw in ("Eau de Parfum", "Eau de Toilette", "Parfum", "Cologne",
                   "Eau de Cologne", "Extrait"):
            if kw.lower() in txt.lower() and len(txt) < 60:
                concentration = kw
                break
        if concentration:
            break

    # Note pyramid
    top, middle, base = [], [], []
    pyramid = soup.find("div", id="pyramid") or soup.find("div", class_=lambda c: c and "pyramid" in c.lower())

    if pyramid:
        sections = pyramid.find_all("div", recursive=False)
        # Fragrantica renders top → heart → base in order
        for i, section in enumerate(sections[:3]):
            header = section.find(["h4", "h3", "h2"])
            header_text = header.get_text(strip=True).lower() if header else ""
            notes = _parse_notes_from_section(section)
            if "top" in header_text or i == 0:
                top = notes
            elif "heart" in header_text or "middle" in header_text or i == 1:
                middle = notes
            elif "base" in header_text or i == 2:
                base = notes
    else:
        # Alternative layout: sections labeled by header text anywhere on page
        for header in soup.find_all(["h3", "h4"]):
            label = header.get_text(strip=True).lower()
            parent = header.find_parent("div")
            if not parent:
                continue
            notes = _parse_notes_from_section(parent)
            if "top" in label:
                top = notes
            elif "heart" in label or "middle" in label:
                middle = notes
            elif "base" in label:
                base = notes

    # Main accords — the coloured bars near the top of the page
    accords = []
    for accord_el in soup.select(".accord-box span, .accord-bar ~ span"):
        txt = accord_el.get_text(strip=True)
        if txt and len(txt) < 40:
            accords.append(txt.lower())
    # Deduplicate while preserving order
    accords = list(dict.fromkeys(accords))

    return {
        "name": name,
        "brand": brand,
        "price_tier": tier,
        "url": url,
        "concentration": concentration,
        "top_notes": "|".join(top),
        "middle_notes": "|".join(middle),
        "base_notes": "|".join(base),
        "main_accords": "|".join(accords[:5]),  # top 5 accords
    }


def scrape(
    brands: list[tuple[str, str]] = BRANDS,
    output_path: str = "data/raw/fragrances_raw.csv",
    max_per_brand: int = 150,
    sleep_min: float = 1.2,
    sleep_max: float = 3.0,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: track already-scraped URLs
    scraped_urls: set[str] = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            scraped_urls = {row["url"] for row in csv.DictReader(f)}
        log.info("Resuming — %d records already in %s", len(scraped_urls), out)

    session = requests.Session()
    write_header = not out.exists() or out.stat().st_size == 0

    with open(out, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        for brand_slug, tier in brands:
            brand_url = f"{BASE_URL}/designers/{brand_slug}.html"
            log.info("Scraping brand: %s (%s)", brand_slug, tier)

            soup = _get(brand_url, session)
            if soup is None:
                continue

            frag_links = _parse_brand_page(soup)
            log.info("  Found %d fragrance links", len(frag_links))

            count = 0
            for rel_url in frag_links:
                if count >= max_per_brand:
                    break
                full_url = BASE_URL + rel_url
                if full_url in scraped_urls:
                    count += 1
                    continue

                time.sleep(random.uniform(sleep_min, sleep_max))
                frag_soup = _get(full_url, session)
                if frag_soup is None:
                    continue

                # The brand name from the URL slug (readable form)
                brand_display = brand_slug.replace("-", " ")
                record = _parse_fragrance_page(frag_soup, full_url, brand_display, tier)
                if record:
                    writer.writerow(record)
                    csvfile.flush()
                    scraped_urls.add(full_url)
                    count += 1
                    log.info("  [%d] %s — %s", count, record["name"], record["concentration"])

    log.info("Done. Output: %s", out)


if __name__ == "__main__":
    scrape()
