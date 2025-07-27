# The criti-scan project  🚀🤓

![logo](https://github.com/SphRbtHyk/criti-scan/blob/24dee13365fda4216f61271b2160c590bec76771/public/logo.jpg)

CritiScan is a tool designed to extract text from PDF-based critical editions—typically scanned or image-based pages that are difficult or impossible to copy and paste due to their complex layout and use of specialized symbols.

The tool detects and isolates the main body of the critical text on each page, applies OCR, and exports the results in two formats:

- Plain text files (easily editable in Word or any text editor)
- Structured JSON files. If specified as such, the verse numbers will be used to structure the output files.

Version 0.0 focuses exclusively on the main text and currently ignores the critical apparatus. Future releases aim to incorporate apparatus parsing and more advanced layout recognition.


## Installation

⚠️⚠️ Tesseract must be installed beforehand (refer to Tesseract documentation for instructions). 
⚠️⚠️ The wanted models for the Tesseract's OCR model must be downloaded from the available ressources, and put in the folder **tessdata**.

Once installed, you can git clone this repository, and install it as a standard Python package:

```bash
git clone git@github.com:SphRbtHyk/criti-scan.git

cd criti-scan

pip install .
```

## Use

CritiScan is usable through an easy to use CLI, that requires as input the path to the PDF to scan, the language of the text, the format of the expected output, as well as its path. Optionally, a user can specify is the text is versified and if the algorithm should try to figure out the verse cut-off of the texts.

```bash
criti-scan input.pdf output.txt \
  --language grc \
  --tessdata-dir /path/to/tessdata
```

For example, you can run the following toy example at the root of thie repository:

```bash
criti-scan tests/test_data/test_pdf.pdf output.txt -l grc -t tessdata/
```


## Incoming!

Incoming work will include the possibility to detect the apparatus and reconstruct variant readings automatically.
