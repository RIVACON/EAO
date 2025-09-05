"""
This is a standard script, available online, to generate a .svg badge file
reading the content of coverage.json.
"""

import json
import os

try:
    with open('coverage.json', 'r') as f:
        data = json.load(f)

    coverage_percentage = int(data['totals']['percent_covered_display'])

    color = "red"
    if coverage_percentage > 70:
        color = "yellow"
    if coverage_percentage > 90:
        color = "green"

    badge_content = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="105" height="20" role="img" aria-label="Coverage: {coverage_percentage}%">
    <title>Coverage: {coverage_percentage}%</title>
    <linearGradient id="s" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <clipPath id="r">
        <rect width="105" height="20" rx="3" fill="#fff"/>
    </clipPath>
    <g clip-path="url(#r)">
        <rect width="70" height="20" fill="#555"/>
        <rect x="70" width="35" height="20" fill="{color}"/>
        <rect width="105" height="20" fill="url(#s)"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="11px">
        <text aria-hidden="true" x="35" y="15" fill="#010101" fill-opacity=".3">coverage</text>
        <text x="35" y="14">coverage</text>
        <text aria-hidden="true" x="87.5" y="15" fill="#010101" fill-opacity=".3">{coverage_percentage}%</text>
        <text x="87.5" y="14">{coverage_percentage}%</text>
    </g>
</svg>
    """

    with open('coverage-badge.svg', 'w') as f:
        f.write(badge_content)

except FileNotFoundError:
    print("Error: coverage.json not found. Please ensure pytest --cov is run first.")