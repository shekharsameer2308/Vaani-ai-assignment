import os
import re
import subprocess
import pypdf

# Read SUBMISSION.md
with open("SUBMISSION.md", "r", encoding="utf-8") as f:
    md_content = f.read()

# Strip markdown header
lines = md_content.splitlines()

# Custom lightweight markdown-to-HTML formatter tailored for 2-page print
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
            # Render table
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
        html_body.append("<hr class='divider'/>")
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

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
    size: letter portrait;
    margin: 0.55in 0.65in;
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 9.2pt;
    line-height: 1.28;
    color: #111;
    margin: 0;
    padding: 0;
  }}
  h1.doc-title {{
    font-size: 14pt;
    font-weight: 700;
    margin: 0 0 6px 0;
    text-align: center;
    border-bottom: 1.5px solid #222;
    padding-bottom: 4px;
  }}
  h2.section-heading {{
    font-size: 10.5pt;
    font-weight: 700;
    margin: 8px 0 3px 0;
    color: #0b2545;
    border-bottom: 0.8px solid #cbd5e1;
    padding-bottom: 1px;
    page-break-after: avoid;
  }}
  h3.sub-heading {{
    font-size: 9.4pt;
    font-weight: 600;
    margin: 4px 0 2px 0;
    color: #1e3a8a;
    page-break-after: avoid;
  }}
  p.paragraph {{
    margin: 2px 0 4px 0;
    text-align: justify;
  }}
  li.list-item, li.ordered-item {{
    margin: 1px 0 1px 16px;
    padding: 0;
  }}
  code {{
    font-family: "SF Mono", Consolas, Menlo, Monaco, monospace;
    font-size: 8.2pt;
    background-color: #f1f5f9;
    padding: 1px 3px;
    border-radius: 2px;
  }}
  table.results-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 4px 0 6px 0;
    font-size: 7.8pt;
    page-break-inside: avoid;
  }}
  table.results-table th, table.results-table td {{
    border: 0.6px solid #94a3b8;
    padding: 2.5px 4px;
    text-align: left;
  }}
  table.results-table th {{
    background-color: #f8fafc;
    font-weight: 600;
  }}
  blockquote.quote-block {{
    border-left: 2.5px solid #64748b;
    margin: 3px 0 3px 12px;
    padding: 1px 8px;
    font-style: italic;
    color: #334155;
  }}
  hr.divider {{
    border: none;
    border-top: 0.6px solid #e2e8f0;
    margin: 6px 0;
  }}
</style>
</head>
<body>
{"".join(html_body)}
</body>
</html>
"""

with open("submission.html", "w", encoding="utf-8") as f:
    f.write(full_html)

# Generate PDF with Chrome headless
chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
cmd = [
    chrome_path,
    "--headless",
    "--disable-gpu",
    "--no-sandbox",
    "--user-data-dir=/tmp/chrome_profile",
    "--print-to-pdf=submission.pdf",
    os.path.abspath("submission.html")
]
subprocess.run(cmd, check=True)

# Check page count
reader = pypdf.PdfReader("submission.pdf")
print(f"Generated submission.pdf successfully. Total pages: {len(reader.pages)}")
