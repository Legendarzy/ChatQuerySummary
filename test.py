import csv
import numpy as np
import re
import faiss
from FlagEmbedding import BGEM3FlagModel,FlagReranker
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
        
    def search(self, query, k=10):
        """
        根据查询向量进行检索
        :param query_vector: 查询语句
        :param k: 要返回的最相似结果的数量
        :return: 最相似的文本及其对应的相似度得分
        """
        # 转换查询向量为 float32 类型
        query_vector = np.array(query, dtype='float32')
        # 对查询向量进行归一化
        faiss.normalize_L2(query_vector)
        # 在 Faiss 索引中进行搜索
        distances, indices = self.index.search(query_vector, k)
        # 获取最相似的文本
        results = [self.texts[i] for i in indices[0]]
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
                            temp_str=str(row['StrTime'])
            if len(temp_str)>0:
                StrContent.append(temp_str)
        self.StrContent=StrContent
    def __getitem__(self,id):
        return self.StrContent[self.batchSize*id:self.batchSize*(id+1)]
    def __len__(self):
        return len(self.StrContent)//self.batchSize
class reRanker:
    def __init__(self):
        self.reranker = FlagReranker('bge-reranker-v2-m3', use_fp16=True)
    def reorder(self,list1, list2,k):
        # 首先将两个列表组合成一个包含元组的列表，每个元组由 list1 和 list2 对应位置的元素组成
        combined = list(zip(list1, list2))
        # 对组合后的列表按照 list2 中的元素进行降序排序
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        # 创建一个空字典用于存储最终结果
        result = {}
        # 遍历排序后的组合列表
        for item1, item2 in sorted_combined[:k]:
            # 将 list1 中的元素作为键，list2 中对应的元素作为值添加到字典中
            result[item1] = item2
        return result
    def rerank(self,query,answer:list,k):
        scores = self.reranker.compute_score([[query,p] for p in answer], normalize=True)
        result=self.reorder(answer,scores,k)
        return result


if __name__ == "__main__":
    data=dataset('file.csv')
    model = BGEM3FlagModel('bge_m3',use_fp16=True)
    retrieval = TextVectorRetrieval(dim=1024)
    reRanker_model=reRanker()
    for d in tqdm(data):
        if len(d)==0:
            break
        vector=model.encode(d,max_length=120)['dense_vecs']
        retrieval.add_vectors(d, vector)
    while 1:
        # 示例查询向量
        query = input('输入你想查询的问题\n')
        query_vector=model.encode([query],max_length=120)['dense_vecs']
        # 进行检索，返回最相似的 2 个结果
        results = retrieval.search(query_vector, k=50)
        result=reRanker_model.rerank(query,results,5)
        for a in result:
            print(f'得分:{result[a]}    文本: {a}')
        


