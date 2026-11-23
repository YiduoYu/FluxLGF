"""Build report.html from report_body.html: {{FIG:name.png}} -> embedded image, {{TABLES}} -> results/tables.html."""

import base64
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>方法二对比报告</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#e6e5e0; --accent:#2a78d6; --card:#ffffff; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#1a1a19; --ink:#f0efec; --muted:#c3c2b7;
  --line:#3a3a37; --accent:#3987e5; --card:#232322; }} }}
:root[data-theme="dark"] {{ --bg:#1a1a19; --ink:#f0efec; --muted:#c3c2b7; --line:#3a3a37; --accent:#3987e5; --card:#232322; }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.65 -apple-system, "PingFang SC", "Helvetica Neue", sans-serif;
  max-width:1100px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:26px; margin-bottom:4px; }} h2 {{ margin-top:36px; border-bottom:1px solid var(--line); padding-bottom:4px; }}
h3 {{ margin-top:24px; }} .sub {{ color:var(--muted); }}
img {{ max-width:100%; background:#fff; border:1px solid var(--line); border-radius:6px; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; font-size:12.5px; margin:8px 0 16px; font-variant-numeric:tabular-nums; }}
th, td {{ border:1px solid var(--line); padding:4px 7px; text-align:center; vertical-align:top; }}
th {{ background:var(--card); }}
.box {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:6px;
  padding:10px 16px; margin:14px 0; }}
code {{ font-size:13px; }}
</style></head><body>
{body}
</body></html>"""


def embed(match):
    data = base64.b64encode((RES / match.group(1)).read_bytes()).decode()
    return f'<img alt="{match.group(1)}" src="data:image/png;base64,{data}">'


def main():
    body = (HERE / "report_body.html").read_text()
    body = re.sub(r"\{\{FIG:([\w.]+)\}\}", embed, body)
    body = body.replace("{{TABLES}}", '<div class="scroll">' + (RES / "tables.html").read_text() + "</div>")
    (HERE / "report.html").write_text(PAGE.format(body=body))


if __name__ == "__main__":
    main()
