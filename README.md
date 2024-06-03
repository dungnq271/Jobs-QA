# RAG for multiple documents

## Demo
![](./annotations/output_v2.gif)

## Supported Documents
- [x] .csv (currenly supported tables in [jobs_posted_v1.csv](./documents/jobs_posted_v1.csv) format only)
- [x] .pdf, .pptx

## Setup
```
pip install -r requirements.txt
```

## Run docker client
```
./scripts/qdrant.sh
```

## Run
**Open 1 tab and run**:

```
./scripts/api.sh
```

**Wait for api startup complete and run in another tab**:
```
./scripts/app.sh {PORT_NUMBER}
```
