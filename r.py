import subprocess

# Path to your Pandoc executable and input/output files
pandocPath = r"C:\Users\pwisz\AppData\Local\Pandoc\pandoc.exe"
wkhtmltopdfPath = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
mdFile = "proj.md"
htmlFile = "proj.html"
pdfFile = "proj.pdf"

cssFile = "--css=proj.css"
title = "--metadata title='proj'"

# Build the command as a list of arguments
cmd1 = [pandocPath, mdFile, "-o", htmlFile, cssFile, title]
cmd2 = [wkhtmltopdfPath, htmlFile, pdfFile]

# Run the command
subprocess.run(cmd1)
subprocess.run(cmd2)