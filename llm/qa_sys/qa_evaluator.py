import json
import re
from helper.helper import get_completion_from_messages

class QAEvaluator:
    """评估问答系统的输出质量"""

    def __init__(self, delimiter="####", debug=False):
        """
        初始化评估器

        Args:
            delimiter (str): 分隔符
            debug (bool): 是否启用调试模式
        """
        self.delimiter = delimiter
        self.debug = debug
        self.product_catalog = self._load_product_catalog()

    def _load_product_catalog(self):
        """加载产品目录"""
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
            ],
            "游戏机和配件": [
                "GameSphere X", "ProGamer Controller", "GameSphere Y",
                "ProGamer Racing Wheel", "GameSphere VR Headset"
            ],
            "音频设备": [
                "AudioPhonic 降噪耳机", "WaveSound 蓝牙音箱",
                "AudioPhonic 真无线耳机", "WaveSound 音箱", "AudioPhonic 唱片机"
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
        您将提供客户服务查询。
        客户服务查询将用{self.delimiter}字符分隔。
        输出一个 Python 列表，列表中的每个对象都是 Json 对象，每个对象的格式如下：
            'category': <电脑和笔记本电脑, 智能手机和配件, 电视和家庭影院系统, 
        游戏机和配件, 音频设备, 相机和摄像机中的一个>,
        以及
            'products': <必须在下面允许的产品中找到的产品列表>
        
        其中类别和产品必须在客户服务查询中找到。
        如果提到了一个产品，它必须与下面允许的产品列表中的正确类别关联。
        如果没有找到产品或类别，输出一个空列表。
        
        不要输出任何不是 JSON 格式的额外文本。
        输出请求的 JSON 后，不要写任何解释性的文本。
        
        根据产品名称和产品类别与客户服务查询的相关性，列出所有相关的产品。
        不要从产品的名称中假设任何特性或属性，如相对质量或价格。
        
        允许的产品以 JSON 格式提供。
        每个项目的键代表类别。
        每个项目的值是该类别中的产品列表。
        允许的产品：{self.product_catalog}
        """

        few_shot_user_1 = """我想要最贵的电脑。你推荐哪款？"""
        few_shot_assistant_1 = """ 
        [{'category': '电脑和笔记本电脑', 'products': ['TechPro 超极本', 'BlueWave 游戏笔记本电脑', 'PowerLite 可转换笔记本电脑', 'TechPro 台式电脑', 'BlueWave Chromebook']}]
        """

        few_shot_user_2 = """我想要最便宜的电脑。你推荐哪款？"""
        few_shot_assistant_2 = """ 
        [{'category': '电脑和笔记本电脑', 'products': ['TechPro 超极本', 'BlueWave 游戏笔记本电脑', 'PowerLite 可转换笔记本电脑', 'TechPro 台式电脑', 'BlueWave Chromebook']}]
        """

        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"{self.delimiter}{few_shot_user_1}{self.delimiter}"},
            {'role': 'assistant', 'content': few_shot_assistant_1},
            {'role': 'user', 'content': f"{self.delimiter}{few_shot_user_2}{self.delimiter}"},
            {'role': 'assistant', 'content': few_shot_assistant_2},
            {'role': 'user', 'content': f"{self.delimiter}{user_query}{self.delimiter}"},
        ]

        return get_completion_from_messages(messages)

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
            # 首先尝试直接解析
            return self._parse_json_response(response)
        except Exception as e:
            if self.debug:
                print(f"警告: 无法解析JSON响应: {response}")
            return []

    def _parse_json_response(self, response):
        """
        解析可能包含markdown代码块的JSON响应
        
        Args:
            response (str): 可能包含JSON的响应
            
        Returns:
            list: 解析后的JSON对象
        """
        import json
        import re
        
        # 1. 首先尝试直接解析整个响应
        try:
            # 将单引号替换为双引号以确保JSON格式正确
            cleaned_response = response.replace("'", "\"")
            return json.loads(cleaned_response)
        except:
            pass
        
        # 2. 尝试去除markdown代码块标记
        try:
            # 移除 ```python, ```json 等标记
            markdown_removed = re.sub(r'```\w*\n', '', response)
            markdown_removed = markdown_removed.replace('```', '')
            cleaned_response = markdown_removed.replace("'", "\"")
            return json.loads(cleaned_response)
        except:
            pass
            
        # 3. 尝试查找所有JSON数组并合并
        try:
            all_json_lists = []
            json_lists = re.findall(r'\[(.*?)\]', response.replace('\n', ' '), re.DOTALL)
            
            if not json_lists:
                return []
                
            # 只取第一个找到的JSON数组
            combined_json = "[" + json_lists[0] + "]"
            combined_json = combined_json.replace("'", "\"")
            return json.loads(combined_json)
        except:
            pass
            
        # 4. 最后的尝试，使用更复杂的正则表达式
        try:
            pattern = r'\[\s*{.*?}\s*(?:,\s*{.*?}\s*)*\]'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_str = match.group(0).replace("'", "\"")
                return json.loads(json_str)
        except:
            pass
        
        # 如果所有尝试都失败，返回空列表
        return []

    def evaluate_response_with_ideal(self, response, ideal_answer, debug=None):
        """
        评估回复是否与理想答案匹配

        Args:
            response (str): LLM的回复
            ideal_answer (dict): 理想的答案，格式为{类别: set(产品列表)}
            debug (bool): 是否打印调试信息，默认使用实例的debug设置

        Returns:
            float: 正确率(0-1)
        """
        if debug is None:
            debug = self.debug

        if debug:
            print("回复：")
            print(response)

        # 判断response是否已经是列表，如果是字符串则需要解析
        if isinstance(response, str):
            try:
                # json.loads() 只能解析双引号，因此此处将单引号替换为双引号
                json_like_str = response.replace("'", '"')

                # 解析为一系列的字典
                l_of_d = json.loads(json_like_str)
            except json.JSONDecodeError:
                if debug:
                    print(f"警告: 无法解析JSON响应: {response}")
                # 尝试正则表达式提取
                try:
                    json_match = re.search(r'(\[.*\])', response.replace('\n', ''), re.DOTALL)
                    if json_match:
                        l_of_d = json.loads(json_match.group(1))
                    else:
                        return 0
                except:
                    return 0
        else:
            l_of_d = response

        # 当响应为空，即没有找到任何商品时
        if l_of_d == [] and ideal_answer == []:
            return 1

        # 另外一种异常情况是，标准答案数量与回复答案数量不匹配
        elif l_of_d == [] or ideal_answer == []:
            return 0

        # 统计正确答案数量
        correct = 0

        if debug:
            print("l_of_d is")
            print(l_of_d)

        # 对每一个问答对
        for d in l_of_d:
            # 获取产品和目录
            cat = d.get('category')
            prod_l = d.get('products')
            # 有获取到产品和目录
            if cat and prod_l:
                # convert list to set for comparison
                prod_set = set(prod_l)
                # get ideal set of products
                ideal_cat = ideal_answer.get(cat)
                if ideal_cat:
                    prod_set_ideal = ideal_cat
                else:
                    if debug:
                        print(f"没有在标准答案中找到目录 {cat}")
                        print(f"标准答案: {ideal_answer}")
                    continue

                if debug:
                    print("产品集合：\n", prod_set)
                    print()
                    print("标准答案的产品集合：\n", prod_set_ideal)

                # 查找到的产品集合和标准的产品集合一致
                if prod_set == prod_set_ideal:
                    if debug:
                        print("正确")
                    correct += 1
                else:
                    if debug:
                        print("错误")
                        print(f"产品集合: {prod_set}")
                        print(f"标准的产品集合: {prod_set_ideal}")
                        if prod_set <= prod_set_ideal:
                            print("回答是标准答案的一个子集")
                        elif prod_set >= prod_set_ideal:
                            print("回答是标准答案的一个超集")

        # 计算正确答案数
        if len(l_of_d) > 0:
            pc_correct = correct / len(l_of_d)
        else:
            pc_correct = 0

        return pc_correct

    def create_test_cases(self):
        """
        创建测试用例

        Returns:
            list: 包含测试用例的列表
        """
        return [
            # eg 0
            {'customer_msg': """如果我预算有限，我可以买哪款电视？""",
             'ideal_answer': {
                 '电视和家庭影院系统': set(
                     ['CineView 4K 电视', 'SoundMax 家庭影院', 'CineView 8K 电视', 'SoundMax 声霸', 'CineView OLED 电视']
                 )}
             },

            # eg 1
            {'customer_msg': """我需要一个智能手机的充电器""",
             'ideal_answer': {
                 '智能手机和配件': set(
                     ['MobiTech PowerCase', 'MobiTech 无线充电器', 'SmartX EarBuds']
                 )}
             },

            # eg 2
            {'customer_msg': """你有什么样的电脑""",
             'ideal_answer': {
                 '电脑和笔记本电脑': set(
                     ['TechPro 超极本', 'BlueWave 游戏笔记本电脑', 'PowerLite 可转换笔记本电脑', 'TechPro 台式电脑', 'BlueWave Chromebook']
                 )}
             },

            # eg 3
            {'customer_msg': """告诉我关于smartx pro手机和fotosnap相机的信息，那款DSLR的。\
            另外，你们有哪些电视？""",
             'ideal_answer': {
                 '智能手机和配件': set(['SmartX ProPhone']),
                 '相机和摄像机': set(['FotoSnap DSLR 相机']),
                 '电视和家庭影院系统': set(['CineView 4K 电视', 'SoundMax 家庭影院', 'CineView 8K 电视', 'SoundMax 声霸', 'CineView OLED 电视'])
             }
             },

            # eg 4
            {'customer_msg': """告诉我关于CineView电视，那款8K电视、\
             Gamesphere游戏机和X游戏机的信息。我的预算有限，你们有哪些电脑？""",
             'ideal_answer': {
                 '电视和家庭影院系统': set(['CineView 8K 电视']),
                 '游戏机和配件': set(['GameSphere X']),
                 '电脑和笔记本电脑': set(['TechPro 超极本', 'BlueWave 游戏笔记本电脑', 'PowerLite 可转换笔记本电脑',
                                          'TechPro 台式电脑', 'BlueWave Chromebook'])
             }
             },

            # eg 5
            {'customer_msg': """你们有哪些智能手机""",
             'ideal_answer': {
                 '智能手机和配件': set(['SmartX ProPhone', 'MobiTech PowerCase', 'SmartX MiniPhone',
                                        'MobiTech 无线充电器', 'SmartX EarBuds'])
             }
             },

            # eg 6
            {'customer_msg': """我预算有限。你能向我推荐一些智能手机吗？""",
             'ideal_answer': {
                 '智能手机和配件': set(['SmartX EarBuds', 'SmartX MiniPhone', 'MobiTech PowerCase',
                                        'SmartX ProPhone', 'MobiTech 无线充电器'])
             }
             },

            # eg 7
            {'customer_msg': """有哪些游戏机适合我喜欢赛车游戏的朋友？""",
             'ideal_answer': {
                 '游戏机和配件': set(['GameSphere X', 'ProGamer Controller', 'GameSphere Y',
                                      'ProGamer Racing Wheel', 'GameSphere VR Headset'])
             }
             },

            # eg 8
            {'customer_msg': """送给我摄像师朋友什么礼物合适？""",
             'ideal_answer': {
                 '相机和摄像机': set(['FotoSnap DSLR 相机', 'ActionCam 4K', 'FotoSnap 无反相机',
                                      'ZoomMaster 摄像机', 'FotoSnap 即时相机'])
             }
             },

            # eg 9
            {'customer_msg': """我想要一台热水浴缸时光机""",
             'ideal_answer': []
             }
        ]

    def batch_evaluate(self, qa_system, test_cases=None, sleep_time=0):
        """
        批量评估问答系统在测试用例上的表现

        Args:
            qa_system: 问答系统实例，需要有process_user_message方法
            test_cases (list): 测试用例列表，如果为None则使用内置测试用例
            sleep_time (int): 每次API调用之间的睡眠时间(秒)

        Returns:
            float: 测试用例的平均正确率
        """
        import time

        if test_cases is None:
            test_cases = self.create_test_cases()

        score_accum = 0
        for i, test_case in enumerate(test_cases):
            if sleep_time > 0:
                time.sleep(sleep_time)

            print(f"测试用例 {i}")

            customer_msg = test_case['customer_msg']
            ideal = test_case['ideal_answer']

            # 使用问答系统回答问题
            response, _ = qa_system.process_user_message(customer_msg)

            # 从用户查询中提取产品信息
            extracted_info = self.extract_product_info(customer_msg)

            # 评估提取的信息是否与理想答案匹配
            score = self.evaluate_response_with_ideal(extracted_info, ideal, debug=self.debug)

            print(f"用例 {i} 分数: {score}")
            score_accum += score

        n_examples = len(test_cases)
        fraction_correct = score_accum / n_examples
        print(f"正确比例为 {n_examples}: {fraction_correct}")

        return fraction_correct

    def evaluate_single_query(self, qa_system, customer_msg, ideal_answer=None):
        """
        评估单个查询的结果

        Args:
            qa_system: 问答系统实例，需要有process_user_message方法
            customer_msg (str): 用户问题
            ideal_answer (dict): 理想答案，如果为None则只评估提取的产品信息

        Returns:
            tuple: (提取的产品信息, 问答系统回答, 评分(如果提供了理想答案))
        """
        # 使用问答系统回答问题
        response, _ = qa_system.process_user_message(customer_msg)

        # 从用户查询中提取产品信息
        extracted_info = self.extract_product_info(customer_msg)

        if ideal_answer:
            # 评估提取的信息是否与理想答案匹配
            score = self.evaluate_response_with_ideal(extracted_info, ideal_answer, debug=self.debug)
            return extracted_info, response, score
        else:
            return extracted_info, response

# 简单的测试函数
def main():
    from llm.qa_sys.qa_system import QASystem

    evaluator = QAEvaluator(debug=True)
    qa_system = QASystem(debug=False)

    # 测试单个查询
    customer_msg = "我想了解SmartX手机和FotoSnap DSLR相机的信息。你们有哪些电视？"

    extracted_info, response = evaluator.evaluate_single_query(qa_system, customer_msg)

    print("\n提取的产品信息:")
    print(extracted_info)

    print("\n问答系统回答:")
    print(response)

    # 使用测试用例批量评估
    print("\n批量评估系统性能:")
    evaluator.batch_evaluate(qa_system, sleep_time=0)

if __name__ == "__main__":
    main()