import os
from reportlab.pdfgen import canvas
from rlextra.rml2pdf import rml2pdf

from utils.paths import output, data

# create a directory for the report if it doesn't exist
output_filename = output("fleet_report.pdf").absolute().as_posix()
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

# Template path
template_path = data("rml/fleet_report_template.rml").absolute().as_posix()

with open(template_path, "r", encoding="utf-8") as template_file:
    # Read the RML template
    rml = template_file.read()

rml2pdf.go(rml, output_filename)
