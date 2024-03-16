# RAG for multiple documents

## Supported Documents
- [x] .pdf, .pptx
- [x] .csv
- [] .jpg, .png
  - [x] ocr
  - [] description

## Setup
```
pip install -r requirements.txt
```

Other dependencies:
```
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

## Run
Open 1 tab and run:
`./scripts/api.sh`

Wait for api startup complete and run in another tab:
`./scripts/app.sh`