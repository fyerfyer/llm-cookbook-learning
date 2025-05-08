from helper.helper import get_completion_from_messages

class QueryClassifier:
    def __init__(self, delimiter="####"):
        self.delimiter = delimiter
        self.system_message = self._create_system_message()
    
    def _create_system_message(self):
        return f"""
        你将获得客户服务查询。
        每个客户服务查询都将用{self.delimiter}字符分隔。
        将每个查询分类到一个主要类别和一个次要类别中。
        以 JSON 格式提供你的输出，包含以下键：primary 和 secondary。

        主要类别：计费（Billing）、技术支持（Technical Support）、账户管理（Account Management）或一般咨询（General Inquiry）。

        计费次要类别：
        取消订阅或升级（Unsubscribe or upgrade）
        添加付款方式（Add a payment method）
        收费解释（Explanation for charge）
        争议费用（Dispute a charge）

        技术支持次要类别：
        常规故障排除（General troubleshooting）
        设备兼容性（Device compatibility）
        软件更新（Software updates）

        账户管理次要类别：
        重置密码（Password reset）
        更新个人信息（Update personal information）
        关闭账户（Close account）
        账户安全（Account security）

        一般咨询次要类别：
        产品信息（Product information）
        定价（Pricing）
        反馈（Feedback）
        与人工对话（Speak to a human）
        """
    
    def classify_query(self, user_query, temperature=0):
        """
        对客户查询进行分类，返回主要类别和次要类别
        
        Args:
            user_query (str): 用户的查询文本
            temperature (float): 模型温度参数，控制随机性
            
        Returns:
            dict: 包含分类结果的字典
        """
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": f"{self.delimiter}{user_query}{self.delimiter}"}
        ]
        
        response = get_completion_from_messages(messages, temperature=temperature)
        
        # 尝试解析JSON响应
        try:
            import json
            import re
            
            # 尝试提取JSON部分
            json_match = re.search(r'({.*})', response.replace('\n', ''), re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                result = json.loads(response)
                
            return result
            
        except Exception as e:
            print(f"解析分类结果时出错: {e}")
            print(f"原始响应: {response}")
            return {
                "error": str(e),
                "raw_response": response
            }

def main():
    print("客户查询分类演示:")
    classifier = QueryClassifier()
    
    # 测试例子1
    print("\n测试例子1: 账户管理查询")
    query1 = "我希望你删除我的个人资料和所有用户数据。"
    result1 = classifier.classify_query(query1)
    print(f"分类结果: {result1}")

    # 测试例子2
    print("\n测试例子2: 技术支持查询")
    query2 = "我的应用程序无法登录，总是显示网络错误。"
    result2 = classifier.classify_query(query2)
    print(f"分类结果: {result2}")
    
    # 测试例子3
    print("\n测试例子3: 计费查询")
    query3 = "为什么我上个月被多收费了？我从未订阅高级服务。"
    result3 = classifier.classify_query(query3)
    print(f"分类结果: {result3}")
    
    # 测试例子4
    print("\n测试例子4: 一般咨询")
    query4 = "你们有提供企业版套餐吗？价格是多少？"
    result4 = classifier.classify_query(query4)
    print(f"分类结果: {result4}")

if __name__ == "__main__":
    main()