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

def generate_coverage_badge(coverage_percentage):
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
    with open(os.path.join(folder, 'coverage-badge.svg'), 'w') as f:
        f.write(badge_content)

def generate_tests_badge(status):
    color = "red"
    if status == "passed":
        color = "green"

    badge_content = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="90" height="20" role="img" aria-label="Tests: {status}">
    <title>Tests: {status}</title>
    <linearGradient id="s" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <clipPath id="r">
        <rect width="90" height="20" rx="3" fill="#fff"/>
    </clipPath>
    <g clip-path="url(#r)">
        <rect width="45" height="20" fill="#555"/>
        <rect x="45" width="45" height="20" fill="{color}"/>
        <rect width="90" height="20" fill="url(#s)"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="11px">
            <text aria-hidden="true" x="22.5" y="15" fill="#010101" fill-opacity=".3">tests</text>
            <text x="22.5" y="14">tests</text>
            <text aria-hidden="true" x="67.5" y="15" fill="#010101" fill-opacity=".3">{status}</text>
            <text x="67.5" y="14">{status}</text>
    </g>
</svg>
"""
    with open(os.path.join(folder, 'tests-badge.svg'), 'w') as f:
        f.write(badge_content)

if __name__ == '__main__':
    try:
        with open('coverage.json', 'r') as f:
            data = json.load(f)
        coverage_percentage = int(data['totals']['percent_covered_display'])
        generate_coverage_badge(coverage_percentage)
    except FileNotFoundError:
        print("Warning: coverage.json not found. Skipping coverage badge generation.")

    try:
        with open('test_status.json', 'r') as f:
            test_status_data = json.load(f)
        generate_tests_badge(test_status_data['status'])
    except FileNotFoundError:
        print("Warning: test_status.json not found. Skipping test status badge generation.")