# RAG for multiple documents

## Supported Documents
- [x] .pdf, .pptx
- [x] .csv
- [ ] .jpg, .png
  - [x] ocr
  - [ ] description

## Setup
```
pip install -r requirements.txt
```

Other dependencies (for ocr):
```
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
pip install paddlepaddle paddleocr
```

## Run
### Open 1 tab and run:

V1 (bot with only index query): `./scripts/api_chat.sh`

V2 (multi-tool bot): `./scripts/api_agent.sh`

### Wait for api startup complete and run in another tab:
`./scripts/app.sh`