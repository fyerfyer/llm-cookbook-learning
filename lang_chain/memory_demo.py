import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch, FAISS
from langchain.chains import RetrievalQA

from helper.helper import create_client

# 使用LangChain嵌入组件
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


class CustomChatModel:
    """使用helper通过Qwen实现的自定义聊天模型"""

    def __init__(self, temperature=0.0):
        self.client = create_client()
        self.temperature = temperature

    def call_as_llm(self, prompt):
        """实现类似LangChain ChatModels的call_as_llm方法"""
        messages = [{"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model="qwen-turbo",
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


class DocumentProcessor:
    """处理文档加载和处理"""

    def __init__(self, file_path):
        self.file_path = file_path
        # 修复：为CSVLoader添加编码参数
        self.loader = CSVLoader(file_path=file_path, encoding="utf-8")
        self.docs = None

    def load_documents(self):
        """从文件加载文档"""
        self.docs = self.loader.load()
        return self.docs

    def split_documents(self, chunk_size=1000, chunk_overlap=0):
        """将文档分割成块"""
        if not self.docs:
            self.load_documents()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(self.docs)

    def preview_data(self):
        """显示数据预览"""
        df = pd.read_csv(self.file_path, usecols=[1, 2], encoding="utf-8")  # 这里也添加编码
        return df.head(5)


class EmbeddingEngine:
    """使用LangChain处理嵌入和向量存储"""

    def __init__(self, embedding_model="openai"):
        """使用指定的嵌入模型初始化"""
        if embedding_model == "openai":
            # 用于实际的OpenAI API
            self.embeddings = OpenAIEmbeddings()
        elif embedding_model == "local":
            # 用于本地模型
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"不支持的嵌入模型: {embedding_model}")

    def create_vector_store(self, docs, vector_store_type="docarray"):
        """从文档创建向量存储"""
        if vector_store_type == "docarray":
            return DocArrayInMemorySearch.from_documents(docs, self.embeddings)
        elif vector_store_type == "faiss":
            return FAISS.from_documents(docs, self.embeddings)
        else:
            raise ValueError(f"不支持的向量存储类型: {vector_store_type}")

    def get_sample_embedding(self, text):
        """为文本生成样本嵌入"""
        embed_vector = self.embeddings.embed_query(text)
        return {
            "length": len(embed_vector),
            "first_5_elements": embed_vector[:5]
        }


class DocumentQA:
    """使用嵌入和检索器的文档问答"""

    def __init__(self, vector_store, llm=None):
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever()
        self.llm = llm if llm else CustomChatModel()

    def similarity_search(self, query, k=4):
        """在向量存储上执行相似度搜索"""
        docs = self.vector_store.similarity_search(query, k=k)
        return docs

    def direct_qa(self, query, k=4):
        """使用检索到的文档和LLM进行直接问答"""
        docs = self.similarity_search(query, k=k)
        # 合并文档内容
        doc_content = "".join([doc.page_content for doc in docs])

        # 将查询发送给LLM
        response = self.llm.call_as_llm(f"{doc_content}\n\n问题: {query}")
        return response

    def retrieval_qa_chain(self, query, chain_type="stuff", verbose=False):
        """创建并使用检索QA链"""
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.retriever,
            verbose=verbose
        )
        return qa.run(query)


def create_sample_csv(output_file="lang_chain/data/OutdoorClothingCatalog_sample.csv"):
    """创建用于演示的示例CSV文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 创建示例数据
    data = [
        ["Women's Campside Oxfords", "This ultracomfortable lace-to-toe Oxford boasts a super-soft canvas, thick cushioning, and quality construction for a broken-in feel from the first time you put them on. \n\nSize & Fit: Order regular shoe size. For half sizes not offered, order up to next whole size. \n\nSpecs: Approx. weight: 1 lb.1 oz. per pair. \n\nConstruction: Soft canvas material for a broken-in feel and look. Comfortable EVA innersole with Cleansport NXT® antimicrobial odor control. Vintage hunt, fish and camping motif on innersole. Moderate arch contour of innersole. EVA foam midsole for cushioning and support. Chain-tread-inspired molded rubber outsole with modified chain-tread pattern. Imported. \n\nQuestions? Please contact us for any inquiries."],
        ["Men's Tropical Plaid Short-Sleeve Shirt", "This sun-protection shirt has a UPF 50+ rating – the highest rated sun protection possible, blocking 98% of the sun's harmful rays. \n\nSize & Fit: Slightly Fitted: Softly shapes the body. Falls at hip. \n\nFabric & Care: 100% polyester. Machine wash and dry. \n\nAdditional Features: Wrinkle-resistant. Front and back cape venting lets in cool breezes. Two front bellows pockets. Imported. \n\nSun Protection That Won't Wear Off: Our high-performance fabric provides SPF 50+ sun protection, blocking 98% of the sun's harmful rays."],
        ["Men's Plaid Tropic Shirt, Short-Sleeve", "Our popular tropics shirt is back with the same great features and UPF 50+ sun protection – now in a new, all-recycled quick-dry fabric. \n\nSize & Fit: Slightly Fitted: Softly shapes the body. \n\nFabric & Care: 52% polyester, 48% nylon. Machine wash and dry. \n\nAdditional Features: UPF 50+ rated – the highest rated sun protection possible. Wrinkle-free. Recycled, high-performance QuickDry™ fabric quickly evaporates perspiration. Front and back cape venting lets in cool breezes. Two front bellows pockets. Imported."],
        ["Girls' Ocean Breeze Long-Sleeve Stripe Shirt", "Our moisture-wicking shirt keeps kids comfortable playing by the ocean or in it. \n\nSize & Fit: Slightly Fitted: Softly shapes the body. \n\nFabric & Care: Nylon Lycra®-elastane blend. UPF 50+ rated, the highest rated sun protection possible. Machine wash and dry. \n\nAdditional Features: Quick drying and fade resistant. Sand washes out easily. Holds its shape after multiple washings. Durable seawater- and chlorine-resistant fabric retains its color. Easy to coordinate with any of our girls' swim collection pieces. Imported."],
        ["Men's TropicVibe Shirt, Short-Sleeve", "This Men's sun-protection shirt with built-in UPF 50+ has the lightweight feel you want and the coverage you need when the air is hot and the UV rays are strong. \n\nSize & Fit: Traditional Fit: Relaxed through the chest, sleeve and waist. \n\nFabric & Care: Shell: 71% Nylon, 29% Polyester. Lining: 100% Polyester knit mesh. UPF 50+ rated – the highest rated sun protection possible. Machine wash and dry. \n\nAdditional Features: Wrinkle resistant. Front and back cape venting lets in cool breezes. Two front bellows pockets. Imported. \n\nSun Protection That Won't Wear Off: Our high-performance fabric provides SPF 50+ sun protection, blocking 98% of the sun's harmful rays."],
        ["Recycled Waterhog Dog Mat, Chevron Weave", "Protect your floors from spills and splashing with our super-absorbent and ultra-durable Waterhog mat. \n\nSize & Dimensions: Small: 18\" x 28\". Medium: 22\"W x 35\"L. Large: 27\"W x 45\"L. X-Large: 36\"W x 66\"L. \n\nFeatures: Super-absorbent, quick-drying and resistant to mold and mildew. Holds up to 1½ gallons of water per square yard. Rubber backing keeps mat in place and prevents leaking. Nonskid surface prevents slipping. Stain and fade resistant. \n\nConstruction: 25% recycled polypropylene fiber top. 20% recycled SBR rubber backing. Made in USA."],
        ["Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece", "She'll love the bright colors, ruffles and exciting design of this two-piece. But you'll love that it has a UPF 50+ rating, blocking 98% of the sun's harmful rays. \n\nSize & Fit: Fits close to body for comfort in the water, with room to grow. \n\nFabric & Care: 82% nylon, 18% Lycra® spandex. UPF 50+ rated – the highest rated sun protection possible. Hand wash, line dry. \n\nAdditional Features: Four-way-stretch fabric is sand and chlorine resistant. Holds its shape even after multiple washings. Cute, authentic surfwear styling she'll love to wear. Imported."],
        ["Refresh Swimwear, V-Neck Tankini Contrasts", "Whether you're going for a swim or heading out for a paddleboard session, our tankini provides the comfort, coverage and confidence you need. \n\nSize & Fit: Fits close to the body for comfortable movement without bulk. Falls at high hip. \n\nFabric & Care: 82% nylon, 18% Lycra® spandex. UPF 50+ rated. Hand wash, line dry. \n\nAdditional Features: Empire seam with gathers for a feminine, supportive fit. Lightly padded cups provide shape and coverage. Front-facing seams are figure flattering. Holds its shape and color even after multiple washings. Sand quickly releases from fabric. Side seams measure 6\". Imported."],
        ["EcoFlex 3L Storm Pants", "Our new TEK O2 technology makes our four-season pants more breathable than ever, keeping you comfortable and protected from the weather no matter what Mother Nature throws at you. \n\nSize & Fit: Fit comfortably over base and middle layers. Inseam: 32\". \n\nFabric & Care: 100% nylon with a waterproof breathable laminate membrane. Machine wash and dry. \n\nAdditional Features: Waterproof seams are fully sealed. Articulated knees allow for excellent mobility. Elastic waist with a drawcord for an adjustable fit. Ankle zippers for easy-on, easy-off versatility. Zippered side pockets provide secure storage. Made in USA of imported fabrics."],
    ]

    # 创建DataFrame并保存为CSV - 使用UTF-8编码
    df = pd.DataFrame(data, columns=['name', 'description'])
    df.to_csv(output_file, index=True, encoding='utf-8')  # 添加UTF-8编码
    print(f"示例CSV创建在: {output_file}")
    return output_file


def main():
    """演示文档QA功能的主函数"""
    load_dotenv(find_dotenv())  # 加载环境变量

    # 如果需要则创建示例数据
    file_path = "lang_chain/data/OutdoorClothingCatalog_sample.csv"
    if not os.path.exists(file_path):
        file_path = create_sample_csv(file_path)
    else:
        # 如果文件存在但可能有编码问题，则重新创建
        file_path = create_sample_csv(file_path)

    print("=== 使用LangChain嵌入的文档QA系统 ===\n")

    # 初始化文档处理器
    print("步骤1: 加载和处理文档")
    doc_processor = DocumentProcessor(file_path)
    docs = doc_processor.load_documents()
    print(f"从{file_path}加载了{len(docs)}个文档")

    # 显示示例数据
    df_preview = doc_processor.preview_data()
    print("\n数据预览:")
    print(df_preview)

    # 初始化嵌入引擎（使用本地模型以避免API密钥要求）
    print("\n步骤2: 创建嵌入和向量存储")
    embedding_engine = EmbeddingEngine(embedding_model="local")

    # 创建向量存储
    vector_store = embedding_engine.create_vector_store(docs)
    print("向量存储创建成功")

    # 显示示例嵌入
    sample_text = "防晒衬衫"
    embedding_info = embedding_engine.get_sample_embedding(sample_text)
    print(f"\n'{sample_text}'的示例嵌入:")
    print(f"向量维度: {embedding_info['length']}")
    print(f"前5个元素: {embedding_info['first_5_elements']}")

    # 初始化文档QA
    print("\n步骤3: 设置文档QA系统")
    llm = CustomChatModel()
    doc_qa = DocumentQA(vector_store, llm)

    # 执行相似度搜索
    query = "请推荐一件具有防晒功能的衬衫"
    print(f"\n执行查询的相似度搜索: '{query}'")
    results = doc_qa.similarity_search(query, k=2)
    print(f"找到{len(results)}个相关文档")
    print("\n第一个匹配的摘录:")
    print(results[0].page_content[:200], "...\n")

    # 执行直接QA
    print("步骤4: 使用检索到的文档进行直接QA")
    direct_answer = doc_qa.direct_qa(query)
    print(f"\n查询: {query}")
    print(f"回答:\n{direct_answer}")

    # 执行格式化QA
    formatted_query = "请用markdown表格的方式列出所有具有防晒功能的衬衫，对每件衬衫描述进行总结"
    print("\n步骤5: 使用检索链进行格式化QA")
    print(f"\n查询: {formatted_query}")
    chain_answer = doc_qa.direct_qa(formatted_query)
    print(f"回答:\n{chain_answer}")


if __name__ == "__main__":
    main()