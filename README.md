# RAG for multiple documents
This project was initially developed to address questions related to AI Engineer job postings available through the [Google Jobs API](https://www.google.com/search?channel=fs&client=ubuntu-sn&q=ai+engineer+tuy%E1%BB%83n+d%E1%BB%A5ng&ibp=htl;jobs&sa=X&ved=2ahUKEwjP0KC27L6GAxVRk68BHfeDDIQQudcGKAF6BAgcECw&sxsrf=ADLYWIJqbRuHJU0MULmNk3Q3T-YSAX5v4A:1717397547924). It has since been expanded to support question-answering (QA) functionalities across multiple document formats, including PDF and PPTX files. Additionally, it now features a Google Search tool for enhanced information retrieval.

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
