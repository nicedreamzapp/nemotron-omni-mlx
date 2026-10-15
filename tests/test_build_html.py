"""Unit tests for article/build_html.py (the markdown -> post HTML converter)."""
from __future__ import annotations

import importlib.util

import pytest

from conftest import ROOT


@pytest.fixture(scope="module")
def bh():
    spec = importlib.util.spec_from_file_location("build_html", ROOT / "article" / "build_html.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_inline(bh):
    assert bh.inline("[MLX](https://x.y)") == '<a href="https://x.y">MLX</a>'
    assert bh.inline("**bold** and *it*") == "<strong>bold</strong> and <em>it</em>"
    assert bh.inline("`code`") == "<code>code</code>"
    # a lone * between digits is not emphasis
    assert bh.inline("2*3*4") == "2*3*4"


def test_blocks(bh):
    md = "\n".join(
        [
            "## Title",
            "",
            "first line",
            "second line",
            "",
            "> quoted",
            "> more",
            "",
            "- a",
            "- b",
            "",
            "| h1 | h2 |",
            "|---|---|",
            "| 1 | 2 |",
            "",
            "__VIDEO__",
        ]
    )
    out = bh.convert(md).split("\n\n")
    assert out[0] == "<h2>Title</h2>"
    assert out[1] == "<p>first line second line</p>"
    assert out[2] == "<blockquote><p>quoted more</p></blockquote>"
    assert out[3] == "<ul>\n<li>a</li>\n<li>b</li>\n</ul>"
    assert "<th>h1</th>" in out[4] and "<td>2</td>" in out[4] and "---" not in out[4]
    assert out[5] == bh.VIDEO_HTML


def test_paragraph_stops_at_block_start(bh):
    assert bh.convert("text\n- item").split("\n\n") == ["<p>text</p>", "<ul>\n<li>item</li>\n</ul>"]


def test_committed_html_is_up_to_date(bh):
    """article.html is generated; it should match article.md."""
    md = (ROOT / "article" / "article.md").read_text()
    assert (ROOT / "article" / "article.html").read_text() == bh.convert(md) + "\n"
