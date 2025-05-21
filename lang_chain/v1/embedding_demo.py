import os

from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional

import pandas as pd
import numpy as np

from helper.helper import get_completion, get_completion_from_messages, get_api_key

class TongyiQianwenLLM(LLM):
    """
    通义千问自定义LLM实现
    """
    model_name: str = "qwen-turbo"
    temperature: float = 0.0
    
    @property
    def _llm_type(self) -> str:
        return "tongyi_qianwen"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # 使用helper中的函数调用通义千问
        return get_completion(prompt, model=self.model_name, temperature=self.temperature)
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name}

class TongyiQianwenEmbeddings(Embeddings):
    """
    通义千问嵌入模型实现（模拟版）
    
    由于目前没有直接的通义千问嵌入API调用，
    这个类使用随机向量来模拟嵌入，仅用于演示目的。
    实际应用中应替换为真实的API调用。
    """
    embedding_dim: int = 768  # 模拟的嵌入维度
    
    def __init__(self, random_seed: int = 42):
        """初始化嵌入模型"""
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档列表生成嵌入向量"""
        # 为每个文档生成一个稳定的伪随机向量
        # 在实际应用中，这应该是对文档内容的真实嵌入
        embeddings = []
        for text in texts:
            # 使用文本的哈希值作为随机种子以获得一致的向量
            seed = hash(text) % 10000
            np.random.seed(seed)
            embedding = np.random.uniform(-1, 1, self.embedding_dim).tolist()
            # 归一化向量
            norm = sum(x*x for x in embedding) ** 0.5
            embedding = [x/norm for x in embedding]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """为查询文本生成嵌入向量"""
        # 为查询生成一个伪随机向量
        # 在实际应用中，这应该是对查询内容的真实嵌入
        seed = hash(text) % 10000
        np.random.seed(seed)
        embedding = np.random.uniform(-1, 1, self.embedding_dim).tolist()
        # 归一化向量
        norm = sum(x*x for x in embedding) ** 0.5
        embedding = [x/norm for x in embedding]
        return embedding

# 自定义聊天模型实现
class TongyiQianwenChat:
    """
    通义千问聊天模型的简单封装
    """
    def __init__(self, temperature=0.0, model="qwen-turbo"):
        self.temperature = temperature
        self.model = model
    
    def __call__(self, messages):
        # 转换为helper能理解的格式
        formatted_messages = []
        for msg in messages:
            role = "user"
            if hasattr(msg, "type"):
                if msg.type == "human":
                    role = "user"
                elif msg.type == "ai":
                    role = "assistant"
                elif msg.type == "system":
                    role = "system"
            content = msg.content
            formatted_messages.append({"role": role, "content": content})
        
        # 调用get_completion_from_messages
        content = get_completion_from_messages(
            formatted_messages, 
            model=self.model,
            temperature=self.temperature
        )
        
        # 创建一个简单的响应对象
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(content)

class DocumentLoader:
    """
    文档加载类，用于加载CSV文件
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.loader = CSVLoader(file_path=file_path, encoding="utf-8")
    
    def load_documents(self):
        """加载文档"""
        return self.loader.load()
    
    def preview_data(self):
        """预览数据"""
        data = pd.read_csv(self.file_path)
        return data.head()

class VectorStoreCreator:
    """
    向量存储创建器
    """
    def __init__(self, embeddings_model=None):
        """初始化向量存储创建器"""
        if embeddings_model is None:
            # 使用自定义的通义千问嵌入模型
            self.embeddings = TongyiQianwenEmbeddings()
        else:
            self.embeddings = embeddings_model
    
    def create_from_documents(self, docs):
        """从文档创建向量存储"""
        return DocArrayInMemorySearch.from_documents(docs, self.embeddings)
    
    def create_retriever(self, docs):
        """创建检索器"""
        db = self.create_from_documents(docs)
        return db.as_retriever()

class DocumentQASystem:
    """
    基于文档的问答系统
    """
    def __init__(self, llm=None, temperature=0.0):
        """初始化问答系统"""
        if llm is None:
            self.llm = TongyiQianwenLLM(temperature=temperature)
        else:
            self.llm = llm
        
        self.chat = TongyiQianwenChat(temperature=temperature)
        self.vector_store_creator = VectorStoreCreator()
    
    def direct_qa(self, docs, query):
        """直接问答方法"""
        # 创建向量存储
        db = self.vector_store_creator.create_from_documents(docs)
        
        # 执行相似度搜索
        similar_docs = db.similarity_search(query)
        
        # 合并获得的相似文档内容
        qdocs = "".join([doc.page_content for doc in similar_docs])
        
        # 调用语言模型生成回答
        response = self.llm._call(f"{qdocs}\n问题：{query}")
        
        return response
    
    def qa_with_chat(self, docs, query):
        """使用聊天模型进行问答"""
        # 创建向量存储
        db = self.vector_store_creator.create_from_documents(docs)
        
        # 执行相似度搜索
        similar_docs = db.similarity_search(query)
        
        # 合并获得的相似文档内容
        qdocs = "".join([doc.page_content for doc in similar_docs])
        
        # 创建提示模板
        template_string = """
        根据以下上下文信息回答问题：
        
        {context}
        
        问题：{query}
        """
        
        prompt_template = ChatPromptTemplate.from_template(template_string)
        messages = prompt_template.format_messages(context=qdocs, query=query)
        
        # 调用聊天模型
        response = self.chat(messages)
        return response.content
    
    def qa_with_retrieval_chain(self, docs, query, verbose=False):
        """使用检索问答链进行问答"""
        # 创建检索器
        retriever = self.vector_store_creator.create_retriever(docs)
        
        # 创建检索问答链
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            verbose=verbose
        )
        
        # 运行链
        return qa_chain.run(query)

def main():
    # 设置数据文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "data", "OutdoorClothingCatalog.csv")
    
    # 创建文档加载器并加载文档
    print("加载文档...")
    doc_loader = DocumentLoader(file_path)
    docs = doc_loader.load_documents()
    
    # 预览数据
    print("\n数据预览:")
    data_preview = doc_loader.preview_data()
    print(data_preview)
    
    # 创建问答系统
    print("\n初始化问答系统...")
    qa_system = DocumentQASystem(temperature=0.0)
    
    # 示例查询
    query = "请用markdown表格的方式列出所有具有防晒功能的衬衫，对每件衬衫描述进行总结"
    
    # 使用直接问答方法
    print("\n\n使用直接问答方法:")
    direct_response = qa_system.direct_qa(docs, query)
    print(direct_response)
    
    # 使用聊天模型进行问答
    print("\n\n使用聊天模型进行问答:")
    chat_response = qa_system.qa_with_chat(docs, query)
    print(chat_response)
    
    # 使用检索问答链进行问答
    print("\n\n使用检索问答链进行问答:")
    chain_response = qa_system.qa_with_retrieval_chain(docs, query, verbose=True)
    print(chain_response)

if __name__ == "__main__":
    main()