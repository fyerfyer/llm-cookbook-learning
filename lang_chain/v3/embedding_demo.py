import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch, FAISS
from langchain.chains import RetrievalQA

from helper.helper import create_client

# 使用langchain的嵌入模型
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


class CustomChatModel:
    """使用Qwen模型的自定义聊天模型"""

    def __init__(self, temperature=0.0):
        self.client = create_client()
        self.temperature = temperature

    def call_as_llm(self, prompt):
        """实现类似langchain LLM的调用"""

        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model="qwen-turbo",
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


class DocumentProcessor:
    """处理文档加载和分割"""

    def __init__(self, file_path):
        self.file_path = file_path
        # 添加编码参数到CSVLoader
        self.loader = CSVLoader(file_path=file_path, encoding="utf-8")
        self.docs = None

    def load_documents(self):
        """从文件中加载文档"""
        self.docs = self.loader.load()
        return self.docs

    def split_documents(self, chunk_size=1000, chunk_overlap=0):
        """文档分块"""
        if not self.docs:
            self.load_documents()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(self.docs)

    def preview_data(self):
        """预览数据"""
        df = pd.read_csv(self.file_path, usecols=[1, 2], encoding="utf-8")
        return df.head(5)


class EmbeddingEngine:
    """使用langchain的嵌入模型"""

    def __init__(self, embedding_model="openai"):
        """初始化嵌入模型"""
        if embedding_model == "openai":
            # 使用OpenAI的嵌入模型
            self.embeddings = OpenAIEmbeddings()
        elif embedding_model == "local":
            # 使用本地的嵌入模型
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"不支持的嵌入模型: {embedding_model}")

    def create_vector_store(self, docs, vector_store_type="docarray"):
        """创建向量存储"""
        if vector_store_type == "docarray":
            return DocArrayInMemorySearch.from_documents(docs, self.embeddings)
        elif vector_store_type == "faiss":
            return FAISS.from_documents(docs, self.embeddings)
        else:
            raise ValueError(f"不支持的向量存储类型: {vector_store_type}")

    def get_sample_embedding(self, text):
        """根据文本获取嵌入向量"""
        embed_vector = self.embeddings.embed_query(text)
        return {
            "length": len(embed_vector),
            "first_5_elements": embed_vector[:5]
        }


class DocumentQA:
    """使用LangChain进行文档问答"""

    def __init__(self, vector_store, llm=None):
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever()
        self.llm = llm if llm else CustomChatModel()

    def similarity_search(self, query, k=4):
        """对文档进行相似性搜索"""
        docs = self.vector_store.similarity_search(query, k=k)
        return docs

    def direct_qa(self, query, k=4):
        """使用检索到的文档进行直接问答"""
        docs = self.similarity_search(query, k=k)
        # 合并文档内容
        doc_content = "".join([doc.page_content for doc in docs])

        # 发送请求到LLM
        response = self.llm.call_as_llm(f"{doc_content}\n\n问题: {query}")
        return response

    def retrieval_qa_chain(self, query, chain_type="stuff", verbose=False):
        """创建并使用检索问答链"""
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.retriever,
            verbose=verbose
        )
        return qa.run(query)


def create_sample_csv(output_file="lang_chain/data/OutdoorClothingCatalog_sample.csv"):
    """创建一个示例CSV文件"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    data = [
        ["Women's Campside Oxfords", "This ultracomfortable lace-to-toe Oxford..."],
        ["Men's Tropical Plaid Short-Sleeve Shirt", "This sun-protection shirt has a UPF 50+ rating..."],
        # ...现有示例数据保持不变...
    ]

    df = pd.DataFrame(data, columns=['name', 'description'])
    df.to_csv(output_file, index=True, encoding='utf-8')
    print(f"示例CSV文件已创建：{output_file}")
    return output_file


def main():
    """展示如何使用LangChain进行文档问答"""
    load_dotenv(find_dotenv())

    file_path = "lang_chain/data/OutdoorClothingCatalog_sample.csv"
    if not os.path.exists(file_path):
        file_path = create_sample_csv(file_path)
    else:
        file_path = create_sample_csv(file_path)

    print("=== 基于LangChain嵌入的文档问答演示 ===\n")

    # 初始化文档处理器
    print("步骤1：加载和处理文档")
    doc_processor = DocumentProcessor(file_path)
    docs = doc_processor.load_documents()
    print(f"已从{file_path}加载{len(docs)}个文档")

    # 显示数据预览
    df_preview = doc_processor.preview_data()
    print("\n数据预览：")
    print(df_preview)

    # 初始化嵌入引擎
    print("\n步骤2：创建嵌入和向量存储")
    embedding_engine = EmbeddingEngine(embedding_model="local")

    vector_store = embedding_engine.create_vector_store(docs)
    print("向量存储创建成功")

    # 显示样本嵌入
    sample_text = "防晒衬衫"
    embedding_info = embedding_engine.get_sample_embedding(sample_text)
    print(f"\n文本'{sample_text}'的嵌入示例：")
    print(f"向量维度：{embedding_info['length']}")
    print(f"前5个元素：{embedding_info['first_5_elements']}")

    # 初始化文档问答
    print("\n步骤3：设置文档问答系统")
    llm = CustomChatModel()
    doc_qa = DocumentQA(vector_store, llm)

    # 执行相似度搜索
    query = "请推荐一件具有防晒功能的衬衫"
    print(f"\n执行相似度搜索，查询：'{query}'")
    results = doc_qa.similarity_search(query, k=2)
    print(f"找到{len(results)}个相关文档")
    print("\n第一条匹配结果摘录：")
    print(results[0].page_content[:200], "...\n")

    # 执行直接问答
    print("步骤4：使用检索文档进行直接问答")
    direct_answer = doc_qa.direct_qa(query)
    print(f"\n问题：{query}")
    print(f"回答：\n{direct_answer}")

    # 执行格式化问答
    formatted_query = "请用markdown表格的方式列出所有具有防晒功能的衬衫，对每件衬衫描述进行总结"
    print("\n步骤5：使用检索链进行格式化问答")
    print(f"\n问题：{formatted_query}")
    chain_answer = doc_qa.direct_qa(formatted_query)
    print(f"回答：\n{chain_answer}")


if __name__ == "__main__":
    main()