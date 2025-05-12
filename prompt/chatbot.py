import json
import panel as pn

from helper.helper import create_client, get_completion_from_messages, get_completion_and_token_count

def test_basic_functionality():
    print("=== 测试基本功能 ===")

    print("\n测试1：莎士比亚风格")
    messages = [
        {'role':'system', 'content':'你是一个像莎士比亚一样说话的助手。'},    
        {'role':'user', 'content':'给我讲个笑话'},   
        {'role':'assistant', 'content':'鸡为什么过马路'},   
        {'role':'user', 'content':'我不知道'}  
    ]
    response = get_completion_from_messages(messages, temperature=1)
    print(f"回答：{response}")

    print("\n测试2：友好的聊天机器人")
    messages = [  
        {'role':'system', 'content':'你是个友好的聊天机器人。'},    
        {'role':'user', 'content':'Hi, 我是Isa。'}  
    ]
    response = get_completion_from_messages(messages, temperature=1)
    print(f"回答: {response}")
    
    print("\n测试3：上下文记忆")
    messages = [  
        {'role':'system', 'content':'你是个友好的聊天机器人。'},
        {'role':'user', 'content':'Hi, 我是Isa'},
        {'role':'assistant', 'content': "Hi Isa! 很高兴认识你。今天有什么可以帮到你的吗?"},
        {'role':'user', 'content':'是的，你可以提醒我, 我的名字是什么?'}  
    ]
    response = get_completion_from_messages(messages, temperature=1)
    print(f"回答: {response}")

class OrderBot:
    def __init__(self):
        # 初始化上下文
        self.context = [{'role':'system', 'content':"""
你是订餐机器人，为披萨餐厅自动收集订单信息。
你要首先问候顾客。然后等待用户回复收集订单信息。收集完信息需确认顾客是否还需要添加其他内容。
最后需要询问是否自取或外送，如果是外送，你要询问地址。
最后告诉顾客订单总金额，并送上祝福。

请确保明确所有选项、附加项和尺寸，以便从菜单中识别出该项唯一的内容。
你的回应应该以简短、非常随意和友好的风格呈现。

菜单包括：

菜品：
意式辣香肠披萨（大、中、小） 12.95、10.00、7.00
芝士披萨（大、中、小） 10.95、9.25、6.50
茄子披萨（大、中、小） 11.95、9.75、6.75
薯条（大、小） 4.50、3.50
希腊沙拉 7.25

配料：
奶酪 2.00
蘑菇 1.50
香肠 3.00
加拿大熏肉 3.50
AI酱 1.50
辣椒 1.00

饮料：
可乐（大、中、小） 3.00、2.00、1.00
雪碧（大、中、小） 3.00、2.00、1.00
瓶装水 5.00
"""}]
        self.panels = [] # 存储对话界面的面板
        self.client = create_client()

    def collect_messages(self, text):
        self.context.append({"role": "user", "content": text})
        response = get_completion_from_messages(self.context)

        # 添加响应到机器人上下文
        self.context.append({"role": "assistant", "content": response})
        return response

    def create_order_summary(self):
        messages = self.context.copy()
        messages.append({
            'role': 'system', 
            'content': '''创建上一个食品订单的 json 摘要。\
逐项列出每件商品的价格，字段应该是 1) 披萨，包括大小 2) 配料列表 3) 饮料列表，包括大小 4) 配菜列表包括大小 5) 总价
返回一个纯JSON对象，不要包含任何格式化字符如```或markdown标记'''
        })

        response = get_completion_from_messages(messages, temperature=0)
        
        # 清理JSON字符串，移除可能的代码块标记
        cleaned_response = response
        if "```json" in response:
            cleaned_response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            cleaned_response = response.split("```")[1].split("```")[0].strip()
            
        return cleaned_response

    def setup_ui(self):
        # 设置交互式UI页面
        pn.extension()
        self.inp = pn.widgets.TextInput(value="你好", placeholder='请输入您的订单...')
        self.button_conversation = pn.widgets.Button(name="发送")
        
        # 绑定按钮动作
        def on_click(event):
            prompt = self.inp.value
            self.inp.value = ''
            
            # 添加用户消息到面板
            self.panels.append(
                pn.Row('用户:', pn.pane.Markdown(prompt, width=600))
            )

            # 获取并显示机器人响应
            response = self.collect_messages(prompt)
            self.panels.append(
                pn.Row('机器人:', pn.pane.Markdown(response, width=600, style={'background-color': '#F6F6F6'}))
            )
            
            # 更新面板
            return pn.Column(*self.panels)

        # 连接按钮点击事件
        self.button_conversation.on_click(on_click)
        
        # 创建摘要按钮
        self.button_summary = pn.widgets.Button(name="生成订单摘要")
        
        def on_summary_click(event):
            summary = self.create_order_summary()
            self.panels.append(
                pn.Row('订单摘要:', pn.pane.Markdown(f"```json\n{summary}\n```", width=600, style={'background-color': '#E8F5E9'}))
            )
            return pn.Column(*self.panels)
        
        # 连接摘要按钮点击事件
        self.button_summary.on_click(on_summary_click)
        
        # 创建界面
        dashboard = pn.Column(
            pn.Row(self.inp, self.button_conversation, self.button_summary),
            pn.panel(pn.bind(lambda: pn.Column(*self.panels)), height=500)
        )
        
        return dashboard

def test_chat_dialogue():
    print("\n=== 测试订餐对话 ===")
    bot = OrderBot()
    
    # 模拟一次订餐对话
    responses = []
    responses.append(bot.collect_messages("你好，我想点一份披萨"))
    responses.append(bot.collect_messages("我要一个大号的意式辣香肠披萨"))
    responses.append(bot.collect_messages("加上额外的奶酪和蘑菇"))
    responses.append(bot.collect_messages("再来一杯中杯可乐"))
    responses.append(bot.collect_messages("我要外送"))
    responses.append(bot.collect_messages("我的地址是北京市海淀区中关村大街1号"))
    
    # 打印所有回复
    for i, response in enumerate(responses):
        print(f"回合 {i+1}:")
        print(f"机器人: {response}\n")

    # 创建订单摘要
    print("\n订单摘要:")
    summary = bot.create_order_summary()
    print(summary)
    
    # 尝试解析JSON
    try:
        parsed_json = json.loads(summary)
        print("\nJSON解析成功!")
    except json.JSONDecodeError:
        print("\nJSON格式可能有问题，无法直接解析")

# 主函数
def main():
    print("欢迎使用通义千问聊天机器人!")
    
    # 测试基本功能
    test_basic_functionality()
    
    # 测试订餐对话
    test_chat_dialogue()
    
    # 启动UI界面 (取消注释以运行)
    print("\n启动交互式UI界面...")
    bot = OrderBot()
    dashboard = bot.setup_ui()
    dashboard.servable()
    
    print("\n请通过访问本地服务器使用聊天机器人")

if __name__ == "__main__":
    main()