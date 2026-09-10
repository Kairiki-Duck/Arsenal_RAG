import time

import requests
from bs4 import BeautifulSoup
from pathlib import Path


PLAYERS = {
    "martin_odegaard": "https://en.wikipedia.org/wiki/Martin_Ødegaard",
}



OUTPUT_DIR = Path("data/raw_data/players")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/131.0 Safari/537.36"
    )
}


def fetch_page(url):
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20,
    )

    response.raise_for_status()

    return response.text


def parse_page(html):
    soup = BeautifulSoup(html, "html.parser")

    content = soup.select_one("#mw-content-text")

    if content is None:
        raise RuntimeError("没有找到 Wikipedia 正文")

    for tag in content.select(
        "script, style, table.navbox, div.navbox, div.reflist"
    ):
        tag.decompose()

    title = soup.select_one("#firstHeading")
    title_text = title.get_text(strip=True) if title else "Unknown"

    lines = []

    lines.append(f"# {title_text}")
    lines.append("")

    for element in content.find_all(
        ["h2", "h3", "h4", "p", "li"]
    ):
        text = element.get_text(" ", strip=True)

        if not text:
            continue

        if element.name == "h2":
            lines.append(f"## {text}")

        elif element.name == "h3":
            lines.append(f"### {text}")

        elif element.name == "h4":
            lines.append(f"#### {text}")

        elif element.name == "li":
            lines.append(f"- {text}")

        else:
            lines.append(text)

        lines.append("")

    return "\n".join(lines)


def crawl_player(name, url):
    print(f"\n正在爬取：{name}")
    print(f"URL: {url}")

    try:
        html = fetch_page(url)

        markdown = parse_page(html)

        output_file = OUTPUT_DIR / f"{name}.md"

        output_file.write_text(
            markdown,
            encoding="utf-8",
        )

        print(f"✓ 保存成功：{output_file}")

    except Exception as e:
        print(f"✗ 爬取失败：{name}")
        print(f"  原因：{e}")


def main():

    for name, url in PLAYERS.items():

        crawl_player(name, url)
        time.sleep(1)


if __name__ == "__main__":
    main()