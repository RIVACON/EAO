"""
This is a standard script, available online, to generate a .svg badge file
reading the content of coverage.json.
This script is supposed to be run from the repository root directory,
so that the script is located in .github/badges,
where the output files will also be located.
"""
import json
import os

folder = os.path.join(".github", "badges")

def generate_notebook_badge(status):
    color = "red"
    if status == "passed":
        color = "green"

    badge_content = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="130" height="20" role="img" aria-label="Notebooks: {status}">
  <title>Notebooks: {status}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="130" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="65" height="20" fill="#555"/>
    <rect x="65" width="65" height="20" fill="{color}"/>
    <rect width="130" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="11px">
    <text aria-hidden="true" x="32.5" y="15" fill="#010101" fill-opacity=".3">Notebooks</text>
    <text x="32.5" y="14">Notebooks</text>
    <text aria-hidden="true" x="97.5" y="15" fill="#010101" fill-opacity=".3">{status}</text>
    <text x="97.5" y="14">{status}</text>
  </g>
</svg>
"""
    with open(os.path.join(folder, 'notebook-badge.svg'), 'w') as f:
        f.write(badge_content)

if __name__ == '__main__':
    try:
        with open('notebooks_status.json', 'r') as f:
            notebook_status_data = json.load(f)
        generate_notebook_badge(notebook_status_data['status'])
    except FileNotFoundError:
        print("Warning: notebook_status.json not found. Skipping test status badge generation.")