import subprocess

# Path to your Pandoc executable and input/output files
pandocPath = r"C:\Users\pwisz\AppData\Local\Pandoc\pandoc.exe"
wkhtmltopdfPath = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
mdFile = "proj.md"
htmlFile = "proj.html"
pdfFile = "proj.pdf"

cssFile = "--css=proj.css"
mathjax = "--mathjax"

# Build the command as a list of arguments
cmd1 = [pandocPath, mdFile, "-o", htmlFile, "-s", cssFile, mathjax]
cmd2 = [
    wkhtmltopdfPath, "--enable-local-file-access", 
    "--margin-top", "1in", 
    "--margin-right", "1in", 
    "--margin-bottom", "0.5in", 
    "--margin-left", "1in", 
    "--page-size", "Letter", htmlFile, pdfFile
]

# Run the command
subprocess.run(cmd1)
subprocess.run(cmd2)