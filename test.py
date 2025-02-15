import csv
import numpy as np
import matplotlib.pyplot as plt
import re
import faiss
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class TextVectorRetrieval:
    def __init__(self, dim):
        # 初始化 Faiss 索引，使用 InnerProduct 度量（可用于计算余弦相似度）
        self.index = faiss.IndexFlatIP(dim)
        # 存储文本的列表
        self.texts = []

    def add_vectors(self, texts, vectors):
        """
        添加文本和对应的向量到检索库中
        :param texts: 文本列表
        :param vectors: 向量的二维数组，形状为 (n, dim)
        """
        # 转换向量为 float32 类型，这是 Faiss 要求的
        vectors = np.array(vectors, dtype='float32')
        # 对向量进行归一化，以便使用 InnerProduct 计算余弦相似度
        faiss.normalize_L2(vectors)
        # 添加向量到 Faiss 索引中
        self.index.add(vectors)
        # 将文本添加到文本列表中
        self.texts.extend(texts)
        
    def search(self, query, tokenizer:BertTokenizer,embed_model:BertModel,k=10):
        """
        根据查询向量进行检索
        :param query_vector: 查询语句
        :param k: 要返回的最相似结果的数量
        :return: 最相似的文本及其对应的相似度得分
        """
        query=tokenizer(query,return_tensors='pt')
        query_vector=embed_model(**query)
        # 转换查询向量为 float32 类型
        query_vector = np.array([query_vector.last_hidden_state[0,0].tolist()], dtype='float32')
        # 对查询向量进行归一化
        faiss.normalize_L2(query_vector)
        # 在 Faiss 索引中进行搜索
        print(query_vector.shape)
        distances, indices = self.index.search(query_vector, k)
        # 获取最相似的文本
        results = [(self.texts[i], distances[0][j]) for j, i in enumerate(indices[0])]
        return results
def remove_urls(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+(?:/[^\s]*)?|#:~:text=[^\s]+')
    if text.count('%')>5:
        return None
    return url_pattern.sub('', text).strip()
class dataset:
    def __init__(self,path):
        StrContent=[]
        self.batchSize=8
        with open(path, 'r', encoding='utf-8') as csvfile:
            # 创建一个 DictReader 对象，用于按字典形式读取 CSV 文件
            reader = csv.DictReader(csvfile)
            temp_str=""
            for row in reader:
                if row['Type']=='1':
                    ss=remove_urls(row['StrContent'])
                    if ss:
                        if len(temp_str)<100:
                            temp_str+=ss+'<sep>'
                        else:
                            # 删去最后的<sep>
                            StrContent.append(temp_str[:-1])
                            temp_str=""
            if len(temp_str)>0:
                StrContent.append(temp_str)
        self.StrContent=StrContent
    def __getitem__(self,id):
        return self.StrContent[self.batchSize*id:self.batchSize*(id+1)]
    def __len__(self):
        return len(self.StrContent)//self.batchSize+1


# 示例使用
if __name__ == "__main__":
    data=dataset('file.csv')
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model=BertModel.from_pretrained('bert-base-chinese')
    retrieval = TextVectorRetrieval(dim=768)

    for d in tqdm(data):
        if len(d)==0:
            break
        out = tokenizer(
            # 传入的两个句子
            text=d,
            # 长度大于设置是否截断
            truncation=True,
            # 一律补齐，如果长度不够
            padding='max_length',
            add_special_tokens=True,
            return_tensors='pt',
            max_length=100,
        )
        out=model(**out)
        vector=out.last_hidden_state[:,0].tolist()
        
        retrieval.add_vectors(d, vector)

    # 示例查询向量
    query_vector = '我们平时去哪里吃饭'
    # 进行检索，返回最相似的 2 个结果
    results = retrieval.search(query_vector,tokenizer,model, k=5)
    print("检索结果：")
    for text, score in results:
        print(f"文本: {text}, 相似度得分: {score}")