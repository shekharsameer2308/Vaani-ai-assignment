import os
import re
import subprocess
import pypdf

# Read SUBMISSION.md
with open("SUBMISSION.md", "r", encoding="utf-8") as f:
    md_content = f.read()

# Separate the prompt history appendix (which is in conversation_history.md)
if "### Key Prompt History (Unedited Prompts)" in md_content:
    body_text = md_content.split("### Key Prompt History (Unedited Prompts)")[0].strip()
else:
    body_text = md_content.strip()

# Dedup Table 3 under Retrieval since Table 3 is already fully presented under Section 3 Results
dup_table = """### Table 3: Retrieval settings (at least three)
| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|
| 250 words, top-3, no re-rank | 15 | 80.0% | 0.0% | 20.0% | 485.1 | 205.0 / 473.4 | 52.0 / 66.7 |
| 500 words, top-5, no re-rank | 15 | 80.0% | 0.0% | 20.0% | 663.9 | 205.2 / 253.7 | 63.2 / 96.7 |
| 500 words, top-10, re-rank yes | 15 | 73.3% | 6.7% | 20.0% | 691.4 | 235.1 / 1086.1 | 67.5 / 120.6 |"""

if dup_table in body_text:
    body_text = body_text.replace(dup_table, "*(Sweep metrics presented in Table 3 under Section 3 above)*")

lines = body_text.splitlines()

html_body = []
in_table = False
table_rows = []

def format_inline(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)
    text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)
    return text

for line in lines:
    line_str = line.strip()
    if not line_str:
        if in_table:
            in_table = False
            html_body.append("<table class='results-table'>")
            for idx, row in enumerate(table_rows):
                cols = [c.strip() for c in row.split("|")[1:-1]]
                if idx == 0:
                    html_body.append("<thead><tr>" + "".join(f"<th>{format_inline(c)}</th>" for c in cols) + "</tr></thead><tbody>")
                elif idx == 1 and all(set(c).issubset({"-", ":", " "}) for c in cols):
                    continue
                else:
                    html_body.append("<tr>" + "".join(f"<td>{format_inline(c)}</td>" for c in cols) + "</tr>")
            html_body.append("</tbody></table>")
            table_rows = []
        continue

    if line_str.startswith("|"):
        in_table = True
        table_rows.append(line_str)
        continue
    elif in_table:
        in_table = False
        html_body.append("<table class='results-table'>")
        for idx, row in enumerate(table_rows):
            cols = [c.strip() for c in row.split("|")[1:-1]]
            if idx == 0:
                html_body.append("<thead><tr>" + "".join(f"<th>{format_inline(c)}</th>" for c in cols) + "</tr></thead><tbody>")
            elif idx == 1 and all(set(c).issubset({"-", ":", " "}) for c in cols):
                continue
            else:
                html_body.append("<tr>" + "".join(f"<td>{format_inline(c)}</td>" for c in cols) + "</tr>")
        html_body.append("</tbody></table>")
        table_rows = []

    if line_str.startswith("# "):
        html_body.append(f"<h1 class='doc-title'>{format_inline(line_str[2:])}</h1>")
    elif line_str.startswith("## "):
        html_body.append(f"<h2 class='section-heading'>{format_inline(line_str[3:])}</h2>")
    elif line_str.startswith("### "):
        html_body.append(f"<h3 class='sub-heading'>{format_inline(line_str[4:])}</h3>")
    elif line_str.startswith("- "):
        html_body.append(f"<li class='list-item'>{format_inline(line_str[2:])}</li>")
    elif re.match(r"^\d+\.\s", line_str):
        html_body.append(f"<li class='ordered-item'>{format_inline(line_str.split('.', 1)[1].strip())}</li>")
    elif line_str.startswith("> "):
        html_body.append(f"<blockquote class='quote-block'>{format_inline(line_str[2:])}</blockquote>")
    elif line_str == "---":
        pass
    else:
        html_body.append(f"<p class='paragraph'>{format_inline(line_str)}</p>")

if in_table:
    html_body.append("<table class='results-table'>")
    for idx, row in enumerate(table_rows):
        cols = [c.strip() for c in row.split("|")[1:-1]]
        if idx == 0:
            html_body.append("<thead><tr>" + "".join(f"<th>{format_inline(c)}</th>" for c in cols) + "</tr></thead><tbody>")
        elif idx == 1 and all(set(c).issubset({"-", ":", " "}) for c in cols):
            continue
        else:
            html_body.append("<tr>" + "".join(f"<td>{format_inline(c)}</td>" for c in cols) + "</tr>")
    html_body.append("</tbody></table>")

css = """
  @page {
    size: letter portrait;
    margin: 0.35in 0.42in;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 7.2pt;
    line-height: 1.16;
    color: #111;
    margin: 0;
    padding: 0;
    column-count: 2;
    column-gap: 12px;
  }
  h1.doc-title {
    font-size: 11.5pt;
    font-weight: 700;
    margin: 0 0 2px 0;
    text-align: center;
    border-bottom: 1.2px solid #222;
    padding-bottom: 1.5px;
    column-span: all;
  }
  h2.section-heading {
    font-size: 8.0pt;
    font-weight: 700;
    margin: 3.5px 0 1px 0;
    color: #0b2545;
    border-bottom: 0.5px solid #cbd5e1;
    padding-bottom: 0.5px;
    break-after: avoid;
  }
  h3.sub-heading {
    font-size: 7.4pt;
    font-weight: 600;
    margin: 2px 0 0.5px 0;
    color: #1e3a8a;
    break-after: avoid;
  }
  p.paragraph {
    margin: 1px 0 1.5px 0;
    text-align: justify;
  }
  li.list-item, li.ordered-item {
    margin: 0.5px 0 0.5px 8px;
    padding: 0;
  }
  code {
    font-family: "SF Mono", Consolas, Menlo, Monaco, monospace;
    font-size: 6.6pt;
    background-color: #f1f5f9;
    padding: 0.5px 1.5px;
    border-radius: 2px;
  }
  table.results-table {
    width: 100%;
    border-collapse: collapse;
    margin: 2px 0 2.5px 0;
    font-size: 6.0pt;
    break-inside: avoid;
  }
  table.results-table th, table.results-table td {
    border: 0.4px solid #94a3b8;
    padding: 1px 1.5px;
    text-align: left;
  }
  table.results-table th {
    background-color: #f8fafc;
    font-weight: 600;
  }
  blockquote.quote-block {
    border-left: 1.5px solid #64748b;
    margin: 1.5px 0 1.5px 6px;
    padding: 0.5px 4px;
    font-style: italic;
    color: #334155;
  }
"""

html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(html_body)}</body></html>"

with open("submission_2pages.html", "w", encoding="utf-8") as f:
    f.write(html)

chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
cmd = [
    chrome_path,
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    "--user-data-dir=/tmp/chrome_profile",
    "--print-to-pdf=submission_2pages.pdf",
    os.path.abspath("submission_2pages.html")
]
subprocess.run(cmd, check=True)

reader = pypdf.PdfReader("submission_2pages.pdf")
print(f"Generated submission_2pages.pdf: {len(reader.pages)} pages")
