import subprocess

# Path to your Pandoc executable and input/output files
pandoc = r"C:\Users\pwisz\AppData\Local\Pandoc\pandoc.exe"
wk = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"

mdF = "proj.md"
htmlF = "proj.html"
pdfF = "proj.pdf"
cssF = "--css=proj.css"

# Build the command as a list of arguments
cmd1 = [
    pandoc, "--mathjax", mdF, "-o", htmlF, "-s", cssF
]

cmd2 = [
    wk, "--enable-local-file-access", 
    "--javascript-delay", "2000", # For equation rendering
    "--no-stop-slow-scripts", # For equation rendering
    "--margin-top", "1in", 
    "--margin-right", "1in", 
    "--margin-bottom", "0.5in", 
    "--margin-left", "1in", 
    "--page-size", "Letter", htmlF, pdfF
]
# Run the command
subprocess.run(cmd1)
#subprocess.run(cmd2)