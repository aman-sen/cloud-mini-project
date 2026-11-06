#!/usr/bin/env bash
# Build report/report.pdf from report/report.md (needs pandoc + wkhtmltopdf)
set -euo pipefail
cd "$(dirname "$0")"
pandoc report.md -f markdown -t html5 -s --css style.css --metadata pagetitle="Report" -o /tmp/report.html
wkhtmltopdf --enable-local-file-access --disable-smart-shrinking --zoom 1.3 -q -s A4 -T 14mm -B 14mm -L 14mm -R 14mm /tmp/report.html report.pdf 2>&1 | tail -1 || true
pdfinfo report.pdf | grep Pages
