import pathlib, time, sys, re, os, html
from lxml import etree
import mwparserfromhell as mw

INPUT_PATH = "enwiki-20251001-pages-articles-multistream.xml"
OUTPUT_DIR = "/mnt/8TB_HDD/datasets/Wikipedia/wiki_pages/"

TABLE_RE = re.compile(r"\{\|[\s\S]*?\|\}", re.DOTALL)
REF_SELF_CLOSE_RE = re.compile(r"(?is)<ref\b[^/>]*/>")
REF_BLOCK_RE      = re.compile(r"(?is)<ref\b[^>]*>.*?</ref>")
GALLERY_RE        = re.compile(r"(?is)<gallery\b[^>]*>.*?</gallery>")
URL_RE            = re.compile(r"https?://\S+")
WS_RE             = re.compile(r"\s+")

pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# --- precompiled regexes (fast path) ---
REF_SELF_CLOSE_RE = re.compile(r"(?is)<ref\b[^/>]*/>")
REF_BLOCK_RE      = re.compile(r"(?is)<ref\b[^>]*>.*?</ref>")
GALLERY_RE        = re.compile(r"(?is)<gallery\b[^>]*>.*?</gallery>")
TABLE_RE          = re.compile(r"(?is)\{\|.*?\|\}")  # crude but fast table stripper
HTML_TAG_RE       = re.compile(r"(?is)<[^>]+>")      # remove leftover HTML-ish tags

# External links: [https://… Label] → Label ; [https://…] → ""
EXT_LINK_LABELED_RE = re.compile(r"\[(?:(?:https?|ftp)://[^\s\]]+)\s+([^\]]+)\]")
EXT_LINK_BARE_RE    = re.compile(r"\[(?:(?:https?|ftp)://[^\s\]]+)\]")

# Internal links:
# [[Page|label]] → label
INT_LINK_LABELED_RE = re.compile(r"\[\[[^|\]]+\|([^\]]+)\]\]")
# [[Page#Section]] or [[Page]] → Page (drop fragment)
INT_LINK_SIMPLE_RE  = re.compile(r"\[\[([^\]|#]+)(?:#[^\]]+)?\]\]")

# File/Image links (we usually don't want captions/thumbnails in plain text) → drop
FILE_LINK_RE        = re.compile(r"\[\[(?:File|Image):[^\]]+\]\]", re.IGNORECASE)

# Simple (non-nested) templates: {{...}} → "" ; repeated until no change
SIMPLE_TEMPLATE_RE  = re.compile(r"\{\{[^{}]*\}\}")

URL_RE = re.compile(r"https?://\S+")
WS_RE  = re.compile(r"\s+")

def _looks_hard_wikitext(s: str) -> bool:
    """Heuristic: only use mwparserfromhell if we see signs of nested templates or parser functions."""
    # Two or more opening braces before a closing brace, or '#if', '#switch', etc.
    return ("{{" in s and "}}" in s and s.count("{{") > 1) or ("{{#" in s)

def clean_wikitext(raw: str) -> str:
    if not raw:
        return ""

    s = raw

    # 0) Very fast structural removals before anything else
    s = TABLE_RE.sub(" ", s)
    s = REF_SELF_CLOSE_RE.sub(" ", s)
    s = REF_BLOCK_RE.sub(" ", s)
    s = GALLERY_RE.sub(" ", s)

    # 1) Fast link handling via regex
    s = FILE_LINK_RE.sub(" ", s)                    # drop [[File:...|...]]
    s = EXT_LINK_LABELED_RE.sub(r"\1", s)           # keep label
    s = EXT_LINK_BARE_RE.sub(" ", s)                # drop bare
    s = INT_LINK_LABELED_RE.sub(r"\1", s)           # keep label
    s = INT_LINK_SIMPLE_RE.sub(r"\1", s)            # keep page title

    # 2) Strip simple templates quickly (non-nested) by iterating a few times
    for _ in range(3):  # usually enough to peel nested in layers cheaply
        new_s = SIMPLE_TEMPLATE_RE.sub(" ", s)
        if new_s == s:
            break
        s = new_s

    # 3) If it still looks “hard”, fall back to mwparserfromhell (rare)
    if _looks_hard_wikitext(s):
        code = mw.parse(s)
        # We avoid remove(); rely on strip_code to drop remaining templates/markup
        out = code.strip_code(normalize=True, collapse=True)
    else:
        # Remove any remaining HTML-ish tags
        s = HTML_TAG_RE.sub(" ", s)
        # Unescape entities (e.g., &nbsp;)
        out = html.unescape(s)

    # 4) Final cleanup
    out = URL_RE.sub("", out)         # nuke stray URLs
    out = WS_RE.sub(" ", out).strip() # whitespace collapse
    return out

def slugify(title, maxlen=150):
    title = title.strip()
    title = re.sub(r"\s+", "_", title)
    title = re.sub(r'[\\/:*?"<>|]', "", title)
    title = re.sub(r"_+", "_", title)[:maxlen].strip("._")
    return title or "untitled"

def _local(tag: str) -> str:
    # "{ns}page" -> "page", "page" -> "page"
    return tag.split('}', 1)[-1] if tag.startswith('{') else tag

def _child_text_by_local(parent, name):
    for ch in parent:
        if _local(ch.tag) == name:
            return ch.text or ""
    return ""

def main(xml_path: str):
    context = etree.iterparse(
        xml_path,
        events=("end",),       # fire only on element closes
        huge_tree=True,
        recover=True,
        encoding="utf-8",
    )

    start = time.time()
    count = 0

    for _, elem in context:
        if _local(elem.tag) != "page":
            continue  # ignore everything until we reach </page>

        # <title> and page-level <id>
        title = _child_text_by_local(elem, "title")
        page_id = _child_text_by_local(elem, "id")

        # Construct filename
        filename = f"{slugify(title)}_{page_id}.txt" if page_id else f"{slugify(title)}.txt"
        if os.path.exists(os.path.join(OUTPUT_DIR, filename)):
            continue

        # <revision>/<text>
        text = ""
        revision_el = next((c for c in elem if _local(c.tag) == "revision"), None)
        if revision_el is not None:
            for sub in revision_el.iter():
                if _local(sub.tag) == "text":
                    text = sub.text or ""
                    break

        # Write page
        if len(text) < 1024:
            continue # skip tiny files
        
        out_path = os.path.join(OUTPUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(clean_wikitext(text))

        # Clear memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        count += 1
        if time.time() - start > 10:
            print(f"Processed {count:,} pages so far", file=sys.stderr)
            start = time.time()

    del context
    print(f"Done. Wrote {count:,} pages to '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main(INPUT_PATH)
