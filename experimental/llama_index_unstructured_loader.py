# %%
import os.path as osp
import cv2
import matplotlib.pyplot as plt
from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import UnstructuredElementNodeParser
from unstructured.partition.image import partition_image
from paddleocr import PaddleOCR

# %%
img_path = "../documents/X51005433548.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

# %%
ocr = PaddleOCR(lang='en') # need to run only once to download and load model into memory

# %%
results = ocr.ocr(img_path, cls=False)

# %%
for result in results:
    print(result[-1][0])

# %%
elements = partition_image(img_path)

# %%
for elem in elements:
    print(elem)
    print("\n")

# %%
extracted_text = '\n'.join([elem.text for elem in elements])
print(extracted_text)

# %%
osp.splitext(img_path)

# %%
