mkdir coco
wget http://images.cocodataset.org/zips/unlabeled2017.zip
unzip unlabeled2017.zip -d coco/
wget https://huggingface.co/datasets/openbmb/llava_zh/blob/main/llava_instruct_150k_zh.json