#!/usr/bin/env python3
"""Turn article.md into the plain-HTML post_content nicedreamzwholesale expects.

Matches the shape of the existing posts (see post 4720): bare <p>/<h2>/<table>,
no Gutenberg block comments, and a <figure> for the video.
"""
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_URL = "https://nicedreamzwholesale.com/wp-content/uploads/2026/07/nemotron_demo.mp4"
POSTER = "https://nicedreamzwholesale.com/wp-content/uploads/2026/07/nemotron-poster.jpg"

VIDEO_HTML = f"""<figure class="wp-block-video"><video controls preload="metadata" playsinline poster="{POSTER}" style="width:100%;height:auto;border-radius:10px;">
<source src="{VIDEO_URL}" type="video/mp4">
</video><figcaption>Thirty-five seconds: the model reads a real cart off my own store, on the laptop, with nothing leaving it. The elapsed counter is the real measured latency.</figcaption></figure>"""


def inline(t: str) -> str:
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<em>\1</em>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t


def convert(md: str) -> str:
    out, i = [], 0
    lines = md.split("\n")
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.strip() == "__VIDEO__":
            out.append(VIDEO_HTML)
            i += 1
        elif ln.startswith("## "):
            out.append(f"<h2>{inline(ln[3:].strip())}</h2>")
            i += 1
        elif ln.startswith("> "):
            block = []
            while i < len(lines) and lines[i].startswith("> "):
                block.append(lines[i][2:].strip())
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(block))}</p></blockquote>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            head, body = rows[0], rows[2:]
            t = ["<table>", "<thead><tr>"]
            t += [f"<th>{inline(c)}</th>" for c in head]
            t.append("</tr></thead>")
            t.append("<tbody>")
            for r in body:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t += ["</tbody>", "</table>"]
            out.append("\n".join(t))
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:].strip())}</li>")
                i += 1
            out.append("<ul>\n" + "\n".join(items) + "\n</ul>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||>|- |__VIDEO__)", lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n\n".join(out)


if __name__ == "__main__":
    md = (ROOT / "article.md").read_text()
    (ROOT / "article.html").write_text(convert(md) + "\n")
    print(f"✓ article.html — {len(convert(md))} chars")
