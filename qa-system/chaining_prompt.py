import json
import re

from helper.helper import get_completion_from_messages

class BasePromptHandler:
    """基础提示处理器类"""
    def __init__(self, delimiter="####"):
        self.delimiter = delimiter

    def process(self, input_text, temperature=0):
        """
        处理输入文本

        Args:
            input_text (str): 输入文本
            temperature (float): 模型温度参数

        Returns:
            str: 处理结果
        """
        messages = self._create_messages(input_text)
        response = get_completion_from_messages(messages, temperature=temperature)
        return self._process_response(response)

    def _create_messages(self, input_text):
        """
        创建消息列表

        Args:
            input_text (str): 输入文本

        Returns:
            list: 消息列表
        """
        return [
            {"role": "system", "content": self._create_system_message()},
            {"role": "user", "content": f"{self.delimiter}{input_text}{self.delimiter}"}
        ]

    def _create_system_message(self):
        """
        创建系统消息

        Returns:
            str: 系统消息
        """
        raise NotImplementedError("子类必须实现此方法")

    def _process_response(self, response):
        """
        处理响应

        Args:
            response (str): 模型响应

        Returns:
            任意类型: 处理后的响应
        """
        return response


class ProductCategoryExtractor(BasePromptHandler):
    """从用户查询中提取产品和类别"""

    def _create_system_message(self):
        return f"""
        您将获得客户服务查询。
        客户服务查询将使用{self.delimiter}字符作为分隔符。
        请仅输出一个可解析的Python列表，列表每一个元素是一个JSON对象，每个对象具有以下格式：
        'category': <类别名称>,
        'products': <产品名称列表>

        类别和产品必须在客户服务查询中找到。
        如果未找到任何产品或类别，则输出一个空列表。
        除了列表外，不要输出其他任何信息！

        允许的类别和产品：

        电脑和笔记本类别:
        - TechPro 超极本
        - BlueWave 游戏本
        - PowerLite Convertible
        - TechPro Desktop
        - BlueWave Chromebook

        智能手机和配件类别:
        - SmartX ProPhone
        - MobiTech PowerCase
        - SmartX MiniPhone
        - MobiTech Wireless Charger
        - SmartX EarBuds

        电视和家庭影院系统类别:
        - CineView 4K TV
        - SoundMax Home Theater
        - CineView 8K TV
        - SoundMax Soundbar
        - CineView OLED TV

        相机和摄像机类别:
        - FotoSnap DSLR Camera
        - ActionCam 4K
        - FotoSnap Mirrorless Camera
        - ZoomMaster Camcorder
        - FotoSnap Instant Camera
            
        只输出对象列表，不包含其他内容。
        """

    def _process_response(self, response):
        """将响应转换为Python对象"""
        try:
            # 将单引号替换为双引号
            response = response.replace("'", "\"")
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"警告: 无法解析JSON响应: {response}")
            # 尝试正则表达式提取
            try:
                json_match = re.search(r'(\[.*\])', response.replace('\n', ''), re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
            except:
                pass
            return []


class ProductInfoRetriever:
    """检索产品信息"""

    def __init__(self):
        self.products_data = self._load_product_data()

    def _load_product_data(self):
        """加载产品数据"""
        # 简化示例数据
        return {
            "TechPro 超极本": {
                "名称": "TechPro 超极本",
                "类别": "电脑和笔记本",
                "价格": 799.99,
                "描述": "一款时尚轻便的超极本，适合日常使用。"
            },
            "BlueWave 游戏本": {
                "名称": "BlueWave 游戏本",
                "类别": "电脑和笔记本",
                "价格": 1199.99,
                "描述": "一款高性能的游戏笔记本电脑，提供沉浸式体验。"
            },
            "SmartX ProPhone": {
                "名称": "SmartX ProPhone",
                "类别": "智能手机和配件",
                "价格": 899.99,
                "描述": "一款拥有先进摄像功能的强大智能手机。"
            },
            "FotoSnap DSLR Camera": {
                "名称": "FotoSnap DSLR Camera",
                "类别": "相机和摄像机",
                "价格": 599.99,
                "描述": "使用这款多功能的单反相机，捕捉惊艳的照片和视频。"
            },
            "CineView 4K TV": {
                "名称": "CineView 4K TV",
                "类别": "电视和家庭影院系统",
                "价格": 599.99,
                "描述": "一款色彩鲜艳、智能功能丰富的惊艳4K电视。"
            }
        }

    def get_info_by_name(self, product_name):
        """根据产品名称获取信息"""
        return self.products_data.get(product_name)

    def get_info_by_category(self, category_name):
        """根据类别获取信息"""
        return [product for product in self.products_data.values()
                if product["类别"] == category_name]

    def retrieve_info(self, extraction_results):
        """根据提取结果检索产品信息"""
        retrieved_info = []

        for item in extraction_results:
            if "category" in item and item["category"]:
                category_products = self.get_info_by_category(item["category"])
                retrieved_info.extend(category_products)

            if "products" in item and item["products"]:
                for product_name in item["products"]:
                    product_info = self.get_info_by_name(product_name)
                    if product_info:
                        retrieved_info.append(product_info)

        return retrieved_info


class ResponseGenerator(BasePromptHandler):
    """生成最终响应"""

    def __init__(self, user_query, product_info):
        super().__init__()
        self.user_query = user_query
        self.product_info = product_info

    def _create_messages(self, _=None):
        """创建消息列表"""
        product_info_str = json.dumps(self.product_info, ensure_ascii=False, indent=2)

        return [
            {"role": "system", "content": self._create_system_message()},
            {"role": "user", "content": self.user_query},
            {"role": "assistant", "content": f"我找到了以下相关产品信息:\n{product_info_str}"}
        ]

    def _create_system_message(self):
        return """
        你是一位客服助理。基于提供的产品信息，用友好的方式回答用户查询。
        保持回复简洁明了，只使用提供的产品信息回答。
        如果没有相关信息，请礼貌地告知用户你没有这方面的信息。
        """

    def generate(self, temperature=0):
        """生成响应"""
        messages = self._create_messages()
        return get_completion_from_messages(messages, temperature=temperature)


class PromptChain:
    """提示链类，将多个提示处理器链接在一起"""

    def __init__(self):
        self.extractor = ProductCategoryExtractor()
        self.retriever = ProductInfoRetriever()

    def process(self, user_query, temperature=0, debug=False):
        """
        处理用户查询

        Args:
            user_query (str): 用户查询
            temperature (float): 模型温度参数
            debug (bool): 是否打印调试信息

        Returns:
            str: 最终响应
        """
        # 步骤1: 提取产品和类别
        if debug:
            print("步骤1: 提取产品和类别...")

        extraction_results = self.extractor.process(user_query, temperature)

        if debug:
            print(f"提取结果: {extraction_results}")

        # 步骤2: 检索产品信息
        if debug:
            print("步骤2: 检索产品信息...")

        product_info = self.retriever.retrieve_info(extraction_results)

        if debug:
            print(f"检索到的产品信息: {product_info}")

        # 步骤3: 生成响应
        if debug:
            print("步骤3: 生成响应...")

        response_generator = ResponseGenerator(user_query, product_info)
        final_response = response_generator.generate(temperature)

        return final_response


def main():
    chain = PromptChain()

    # 测试查询
    print("\n示例1: 产品查询")
    query1 = "请告诉我关于SmartX ProPhone和FotoSnap相机的信息"
    print(f"用户查询: {query1}")
    response1 = chain.process(query1, debug=True)
    print(f"\n最终响应:\n{response1}")

    print("\n\n示例2: 不存在的产品查询")
    query2 = "我想买一个路由器"
    print(f"用户查询: {query2}")
    response2 = chain.process(query2, debug=True)
    print(f"\n最终响应:\n{response2}")

    print("\n\n示例3: 价格比较查询")
    query3 = "BlueWave 游戏本和TechPro超极本哪个更便宜？"
    print(f"用户查询: {query3}")
    response3 = chain.process(query3, debug=True)
    print(f"\n最终响应:\n{response3}")


if __name__ == "__main__":
    main()