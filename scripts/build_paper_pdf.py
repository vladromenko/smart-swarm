"""Render the Smart Swarm manuscript and its archived figures on Homebrain."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

from lxml import etree, html
from markdown_it import MarkdownIt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/PAPER_DRAFT.md"
FIGURES = (
    ("primary-example-f1-views", "Private visual histories in one independent F1 simulation. Panels use the admitted camera scopes."),
    ("primary-test-inspection-controls", "Inspection of F6 pairs and correct direct responses on the resolving controls at eight robots."),
    ("primary-inspection-thresholds", "Coverage, selective risk and confidence bounds from the original eight-robot test."),
    ("primary-matched-compact-controls", "Matched compact communication with identical likelihoods and metadata charges."),
    ("primary-fixed-final-witness-order", "Changes in earlier witness frames on the registered three-robot sources."),
    ("rosmaster-bounded-comparison-effect", "Moving and stationary ROSMASTER views on 17 valid simulated pairs."),
)
PILOT = ROOT / "evidence/learning-pilot-v3"

STYLESHEET = """
@page {size:A4; margin:17mm 21mm 18mm;}
body {font-family:'Liberation Serif'; font-size:11pt; color:#192534;}
h1,h2,h3 {font-family:'Liberation Sans'; color:#17374e; page-break-after:avoid;}
h1 {font-size:29pt; margin-top:5pt; margin-bottom:5pt;}
h2 {font-size:16pt; margin-top:15pt; margin-bottom:5pt;}
h3 {font-size:12pt; margin-top:12pt; margin-bottom:4pt;}
p {font-size:11pt; line-height:122%; margin-top:0; margin-bottom:7pt; orphans:2; widows:2;}
p.kicker {font-family:'Liberation Sans'; font-size:9pt; color:#38627c; font-weight:bold;}
p.subtitle {font-size:14pt; color:#40566a; margin-bottom:15pt;}
p.caption {font-family:'Liberation Sans'; font-size:9pt; color:#40566a; margin-bottom:13pt;}
p.artifact {font-family:'Liberation Sans'; font-size:8pt; line-height:111%; margin-bottom:0;}
li {margin-bottom:4pt;}
a {color:#265674; text-decoration:underline;}
table {border-collapse:collapse; margin-bottom:6pt; page-break-inside:avoid;}
th,td {font-family:'Liberation Sans'; font-size:9pt; border-bottom:1px solid #d5e0e8; padding:3pt;}
th {background-color:#e4edf3; text-align:left;}
pre,code {font-family:'Liberation Mono'; color:#243b4c;}
code {font-size:9pt;}
pre {font-size:8pt; line-height:121%; background-color:#f0f4f7;}
pre code {font-size:8pt;}
.page-start {page-break-before:always;}
"""


def run(command, *, seconds=45):
    try:
        return subprocess.run(command, capture_output=True, check=True, timeout=seconds).stdout
    except subprocess.CalledProcessError as error:
        rendered = error.stderr.decode(errors="replace").strip()
        raise ValueError(f"{command[0]} failed. {rendered}") from error


def inline(parent, tag, text="", **attributes):
    result = etree.SubElement(parent, tag, attributes)
    result.text = text or None
    return result


def annotate_links(body, figures):
    artifacts, external = {}, set()
    for link in body.xpath(".//a[@href]"):
        original = link.attrib["href"]
        route = urlsplit(original)
        if route.scheme in ("http", "https"):
            external.add(original)
        else:
            source = (SOURCE.parent / unquote(route.path)).resolve()
            if route.scheme or route.netloc or not source.is_file() or not source.is_relative_to(ROOT.parent.parent):
                raise ValueError(f"The manuscript references an unavailable local artifact at {original}")
            if source in figures and not route.fragment:
                label, target = figures[source]
            else:
                key = (source, route.fragment)
                label, target = artifacts.setdefault(key, (f"A{len(artifacts) + 1}", f"artifact-{len(artifacts) + 1}"))
            link.set("href", f"#{target}")
            inline(link, "sup", f" {label}")
    return artifacts, external


def graphic(parent, source, number, caption, directory):
    raster = directory / f"figure-{number}"
    run(["pdftocairo", "-png", "-f", "1", "-l", "1", "-r", "220", "-singlefile", str(source), str(raster)])
    image = raster.with_suffix(".png")
    with Image.open(image) as resource:
        width, height = resource.size
    heading = inline(parent, "h3")
    inline(heading, "a", name=f"figure-{number}")
    heading[0].tail = f"Figure {number}"
    line = inline(parent, "p")
    inline(line, "img", src=image.name, width="609", height=str(round(609 * height / width)), alt=caption)
    inline(parent, "p", caption, **{"class": "caption"})


def gallery(parent, directory):
    for index, (name, caption) in enumerate(FIGURES):
        if index % 2 == 0:
            title = "Figures" if index == 0 else "Figures continued"
            inline(parent, "h2", title, **{"class": "page-start"})
        graphic(parent, PILOT / f"{name}.pdf", index + 1, caption, directory)


def artifact_index(parent, artifacts):
    inline(parent, "h2", "Artifact references", **{"class": "page-start"})
    inline(parent, "p", "Paths are relative to the Smart Swarm repository. The Homebrain workspace also holds the indexed raw assets.")
    line = inline(parent, "p", **{"class": "artifact"})
    for (path, fragment), (label, target) in artifacts.items():
        anchor = inline(line, "a", name=target)
        section = f"#{fragment}" if fragment else ""
        anchor.tail = f"{label}   {os.path.relpath(path, ROOT)}{section}"
        inline(line, "br")


def manuscript(directory):
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    body = html.fragment_fromstring(parser.render(SOURCE.read_text()), create_parent="div")
    title = body.xpath("./h1")[0].text_content()
    header = etree.Element("p", {"class": "kicker"})
    header.text = "SMART SWARM RESEARCH DRAFT"
    body.insert(0, header)
    body.xpath("./p")[1].set("class", "subtitle")
    for table in body.xpath(".//table"):
        table.set("width", "100%")
    figures = {(PILOT / f"{name}.pdf").resolve(): (f"Fig. {i}", f"figure-{i}")
               for i, (name, _) in enumerate(FIGURES, 1)}
    artifacts, external = annotate_links(body, figures)
    gallery(body, directory)
    artifact_index(body, artifacts)
    root = etree.Element("html", lang="en")
    head = inline(root, "head")
    inline(head, "meta", charset="UTF-8")
    inline(head, "title", title)
    inline(head, "style", STYLESHEET)
    inline(root, "body").append(body)
    path = directory / "manuscript.html"
    path.write_bytes(b"<!doctype html>\n" + etree.tostring(root, encoding="UTF-8", method="html"))
    destinations = {target for _, target in [*artifacts.values(), *figures.values()]}
    return path, destinations, external


def convert(source, directory, fmt):
    profile = (directory / "office-profile").as_uri()
    run(["libreoffice", f"-env:UserInstallation={profile}", "--headless", "--convert-to", fmt,
         "--outdir", str(directory), str(source)])
    output = directory / f"{source.stem}.{fmt.split(':', 1)[0]}"
    if not output.is_file() or not output.stat().st_size:
        raise ValueError(f"LibreOffice did not produce {output.name}")
    return output


def footer_styles(raw):
    root = etree.fromstring(raw)
    names = root.nsmap

    def qualified(key):
        prefix, tag = key.split(":", 1)
        return f"{{{names[prefix]}}}{tag}"

    def node(parent, tag, attributes=None):
        return etree.SubElement(parent, qualified(tag), {qualified(k): v for k, v in (attributes or {}).items()})

    style = node(root.find("office:styles", names), "style:style", {"style:name": "SwarmFooter", "style:family": "paragraph"})
    node(style, "style:paragraph-properties", {"fo:text-align": "end", "fo:margin-top": "0in", "fo:margin-bottom": "0in"})
    node(style, "style:text-properties", {"fo:font-size": "8pt", "fo:font-family": "Liberation Sans", "fo:color": "#52697b"})
    for master in root.findall(".//style:master-page", names):
        footer = node(master, "style:footer")
        paragraph = node(footer, "text:p", {"text:style-name": "SwarmFooter"})
        paragraph.text = "Smart Swarm · "
        node(paragraph, "text:page-number", {"text:select-page": "current"}).text = "1"
    return etree.tostring(root, encoding="UTF-8", xml_declaration=True)


def intact_tables(raw):
    root = etree.fromstring(raw)
    names = root.nsmap
    for table in root.findall(".//style:table-properties", names):
        table.set(f"{{{names['style']}}}may-break-between-rows", "false")
    return etree.tostring(root, encoding="UTF-8", xml_declaration=True)


def numbered(source, directory):
    output = directory / "paper.odt"
    with ZipFile(source) as incoming, ZipFile(output, "w") as outgoing:
        replacements = {"styles.xml": footer_styles(incoming.read("styles.xml")),
                        "content.xml": intact_tables(incoming.read("content.xml"))}
        for item in incoming.infolist():
            data = replacements[item.filename] if item.filename in replacements else incoming.read(item)
            outgoing.writestr(item, data)
    return output


def verify(path, targets, external):
    info = run(["pdfinfo", str(path)]).decode()
    pages = int(re.search(r"^Pages:\s+(\d+)$", info, re.M)[1])
    destinations = set(re.findall(r'"([^"\n]+)"', run(["pdfinfo", "-dests", str(path)]).decode()))
    urls = {line.split()[-1] for line in run(["pdfinfo", "-url", str(path)]).decode().splitlines()[1:] if line.strip()}
    text = run(["pdftotext", "-layout", str(path), "-"]).decode().split("\f")
    images = [row.split() for row in run(["pdfimages", "-list", str(path)]).decode().splitlines()[2:] if row.strip()]
    embedded = sum(row[2] == "image" and int(row[3]) >= 800 and int(row[4]) >= 500 for row in images)
    if (targets - destinations or urls != external or not re.search(r"^Tagged:\s+yes$", info, re.M)
            or embedded != len(FIGURES)
            or len(text) != pages + 1 or any(f"Smart Swarm · {number}" not in page for number, page in enumerate(text[:-1], 1))):
        raise ValueError("The printed manuscript lost a figure, link, document tag or page number")
    return {"pages": pages, "embedded_figures": embedded, "internal_destinations": len(targets), "external_links": len(external)}


def build(output):
    switches = {"UseTaggedPDF": {"type": "boolean", "value": "true"},
                "ExportBookmarksToPDFDestination": {"type": "boolean", "value": "true"},
                "ReduceImageResolution": {"type": "boolean", "value": "false"},
                "UseLosslessCompression": {"type": "boolean", "value": "true"}}
    with TemporaryDirectory(prefix="smart-swarm-paper-") as temporary:
        directory = Path(temporary)
        source, targets, external = manuscript(directory)
        odt = numbered(convert(source, directory, "odt:writer8"), directory)
        pdf = convert(odt, directory, f"pdf:writer_pdf_Export:{json.dumps(switches)}")
        receipt = verify(pdf, targets, external)
        with NamedTemporaryFile(dir=output.parent, suffix=".pdf", delete=False) as staged:
            path = Path(staged.name)
        try:
            shutil.copyfile(pdf, path)
            path.chmod(0o644)
            path.replace(output)
        finally:
            path.unlink(missing_ok=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/PAPER_DRAFT.pdf")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".pdf":
        parser.error("The output must be a PDF")
    try:
        receipt = build(args.output)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        parser.exit(2, f"Paper export withheld. {error}\n")
    print(json.dumps({"output": str(args.output), **receipt}, indent=2))


if __name__ == "__main__":
    main()
