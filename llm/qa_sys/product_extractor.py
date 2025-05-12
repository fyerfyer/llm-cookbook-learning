import json
import re
from helper.helper import get_completion_from_messages

class ProductExtractor:
    """从用户查询中提取产品和类别信息"""

    def __init__(self, delimiter="####"):
        self.delimiter = delimiter
        self.product_catalog = self._load_product_catalog()

    def _load_product_catalog(self):
        """加载产品目录"""
        # 定义产品目录
        return {
            "电脑和笔记本电脑": [
                "TechPro 超极本", "BlueWave 游戏笔记本电脑", "PowerLite 可转换笔记本电脑",
                "TechPro 台式电脑", "BlueWave Chromebook"
            ],
            "智能手机和配件": [
                "SmartX ProPhone", "MobiTech PowerCase", "SmartX MiniPhone",
                "MobiTech 无线充电器", "SmartX EarBuds"
            ],
            "电视和家庭影院系统": [
                "CineView 4K 电视", "SoundMax 家庭影院", "CineView 8K 电视",
                "SoundMax 声霸", "CineView OLED 电视"
            ],
            "相机和摄像机": [
                "FotoSnap DSLR 相机", "ActionCam 4K", "FotoSnap 无反相机",
                "ZoomMaster 摄像机", "FotoSnap 即时相机"
            ]
        }

    def get_products_and_category(self):
        """获取产品和类别列表"""
        return self.product_catalog

    def find_category_and_product(self, user_query):
        """
        从用户查询中提取产品和类别

        Args:
            user_query (str): 用户查询

        Returns:
            str: 包含提取结果的JSON字符串
        """
        system_message = f"""
        您将获得客户服务查询。
        客户服务查询将使用{self.delimiter}字符作为分隔符。
        请输出一个有效的Python列表，其中包含查询中提到的产品和类别。
        
        产品类别包括:
        电脑和笔记本电脑
        智能手机和配件
        电视和家庭影院系统
        相机和摄像机
        
        产品包括:
        电脑和笔记本电脑类别:
        TechPro 超极本
        BlueWave 游戏笔记本电脑
        PowerLite 可转换笔记本电脑
        TechPro 台式电脑
        BlueWave Chromebook
        
        智能手机和配件类别:
        SmartX ProPhone
        MobiTech PowerCase
        SmartX MiniPhone
        MobiTech 无线充电器
        SmartX EarBuds
        
        电视和家庭影院系统类别:
        CineView 4K 电视
        SoundMax 家庭影院
        CineView 8K 电视
        SoundMax 声霸
        CineView OLED 电视
        
        相机和摄像机类别:
        FotoSnap DSLR 相机
        ActionCam 4K
        FotoSnap 无反相机
        ZoomMaster 摄像机
        FotoSnap 即时相机
        
        输出格式应为:
        [
            {{"category": "类别名称", "products": ["产品1", "产品2", ...]}}
        ]
        
        如果产品或类别不在上面的列表中，则不要包含它们。
        如果没有找到产品或类别，则返回一个空列表。
        仅包含在客户查询中明确提到的产品或类别。
        """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{self.delimiter}{user_query}{self.delimiter}"}
        ]

        response = get_completion_from_messages(messages)
        return response

    def extract_product_info(self, user_query):
        """
        提取产品信息并转换为Python对象

        Args:
            user_query (str): 用户查询

        Returns:
            list: 提取的产品和类别列表
        """
        response = self.find_category_and_product(user_query)

        # 将响应转换为Python对象
        try:
            # 将单引号替换为双引号以确保JSON格式正确
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

    def get_product_information(self, category_and_product_list):
        """
        生成产品信息描述

        Args:
            category_and_product_list (list): 类别和产品列表

        Returns:
            str: 产品信息描述
        """
        # 这里可以添加更详细的产品信息
        product_info = {
            "TechPro 超极本": {
                "名称": "TechPro 超极本",
                "类别": "电脑和笔记本电脑",
                "品牌": "TechPro",
                "型号": "TP-UB100",
                "保修期": "1年",
                "评分": 4.5,
                "特点": "13.3英寸显示屏，8GB RAM，256GB SSD，Intel Core i5处理器",
                "描述": "一款适用于日常使用的时尚轻便的超极本。",
                "价格": 799.99
            },
            "BlueWave 游戏笔记本电脑": {
                "名称": "BlueWave 游戏笔记本电脑",
                "类别": "电脑和笔记本电脑",
                "品牌": "BlueWave",
                "型号": "BW-GL200",
                "保修期": "2年",
                "评分": 4.7,
                "特点": "15.6英寸显示屏，16GB RAM，512GB SSD，NVIDIA GeForce RTX 3060",
                "描述": "一款高性能的游戏笔记本电脑，提供沉浸式体验。",
                "价格": 1199.99
            },
            "PowerLite 可转换笔记本电脑": {
                "名称": "PowerLite 可转换笔记本电脑",
                "类别": "电脑和笔记本电脑",
                "品牌": "PowerLite",
                "型号": "PL-CV300",
                "保修期": "1年",
                "评分": 4.3,
                "特点": "14英寸触摸屏，8GB RAM，256GB SSD，360度铰链",
                "描述": "一款多功能可转换笔记本电脑，具有响应触摸屏。",
                "价格": 699.99
            },
            "TechPro 台式电脑": {
                "名称": "TechPro 台式电脑",
                "类别": "电脑和笔记本电脑",
                "品牌": "TechPro",
                "型号": "TP-DT500",
                "保修期": "1年",
                "评分": 4.4,
                "特点": "Intel Core i7处理器，16GB RAM，1TB HDD，NVIDIA GeForce GTX 1660",
                "描述": "一款功能强大的台式电脑，适用于工作和娱乐。",
                "价格": 999.99
            },
            "BlueWave Chromebook": {
                "名称": "BlueWave Chromebook",
                "类别": "电脑和笔记本电脑",
                "品牌": "BlueWave",
                "型号": "BW-CB100",
                "保修期": "1年",
                "评分": 4.1,
                "特点": "11.6英寸显示屏，4GB RAM，32GB eMMC，Chrome OS",
                "描述": "一款紧凑而价格实惠的Chromebook，适用于日常任务。",
                "价格": 249.99
            },
            "SmartX ProPhone": {
                "名称": "SmartX ProPhone",
                "类别": "智能手机和配件",
                "品牌": "SmartX",
                "型号": "SX-PP10",
                "保修期": "1年",
                "评分": 4.6,
                "特点": "6.1英寸显示屏，12MP双摄像头，5G，128GB存储",
                "描述": "一款功能强大的智能手机，拥有先进的摄影功能。",
                "价格": 899.99
            },
            "FotoSnap DSLR 相机": {
                "名称": "FotoSnap DSLR 相机",
                "类别": "相机和摄像机",
                "品牌": "FotoSnap",
                "型号": "FS-DSLR200",
                "保修期": "1年",
                "评分": 4.4,
                "特点": "24.2MP，1080p视频，3英寸LCD，可更换镜头",
                "描述": "使用这款多功能的单反相机，捕捉惊艳的照片和视频。",
                "价格": 599.99
            },
            "CineView 4K 电视": {
                "名称": "CineView 4K 电视",
                "类别": "电视和家庭影院系统",
                "品牌": "CineView",
                "型号": "CV-4K55",
                "保修期": "2年",
                "评分": 4.8,
                "特点": "55英寸，4K分辨率，HDR，智能电视功能",
                "描述": "一款色彩鲜艳、智能功能丰富的惊艳4K电视。",
                "价格": 599.99
            }
        }

        result = []
        for item in category_and_product_list:
            category = item.get("category", "")
            products = item.get("products", [])

            # 添加类别所有产品
            if category and not products:
                for product_name in self.product_catalog.get(category, []):
                    if product_name in product_info:
                        result.append(product_info[product_name])

            # 添加特定产品
            for product_name in products:
                if product_name in product_info:
                    result.append(product_info[product_name])

        # 将产品信息转换为字符串格式
        if not result:
            return "未找到相关产品信息。"

        info_str = ""
        for product in result:
            info_str += f"产品：{product['名称']}\n"
            info_str += f"类别：{product['类别']}\n"
            info_str += f"品牌：{product['品牌']}\n"
            info_str += f"型号：{product['型号']}\n"
            info_str += f"价格：${product['价格']}\n"
            info_str += f"特点：{product['特点']}\n"
            info_str += f"描述：{product['描述']}\n"
            info_str += f"保修期：{product['保修期']}\n"
            info_str += f"评分：{product['评分']}/5\n\n"

        return info_str.strip()