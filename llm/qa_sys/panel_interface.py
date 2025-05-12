import panel as pn
import sys
import os
from llm.qa_sys.qa_system import QASystem

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def create_qa_interface(debug=False):
    """
    创建问答系统的可视化界面

    Args:
        debug (bool): 是否启用调试模式

    Returns:
        panel.Column: 可视化界面
    """
    # 初始化Panel扩展
    pn.extension()

    # 初始化问答系统
    qa_system = QASystem(debug=debug)

    # 初始化变量
    panels = []  # 收集显示面板
    context = [{'role': 'system', 'content': "You are Service Assistant"}]  # 系统消息

    # 输入控件
    inp = pn.widgets.TextInput(placeholder='请输入您的问题...')
    button_conversation = pn.widgets.Button(name="发送", button_type="primary")

    def collect_messages(event):
        """收集用户消息并生成回答"""
        user_input = inp.value
        if user_input == "":
            return

        inp.value = ''  # 清空输入框

        # 显示用户输入
        panels.append(
            pn.Row('用户:', pn.pane.Markdown(user_input, width=600))
        )

        # 处理用户消息
        nonlocal context
        response, context = qa_system.process_user_message(user_input, context)

        # 显示助手回答
        panels.append(
            pn.Row('助手:', pn.pane.Markdown(response, width=600, styles={'background-color': '#F6F6F6'}))
        )

        # 更新显示
        conversation_panel.objects = [pn.Column(*panels)]

    # 绑定按钮点击事件
    button_conversation.on_click(collect_messages)

    # 创建对话面板
    conversation_panel = pn.Column(pn.Column(*panels))

    # 创建主界面
    dashboard = pn.Column(
        pn.pane.Markdown("# 电子商店客户服务助手"),
        pn.pane.Markdown("请输入您的问题，我会尽力帮助您。"),
        pn.Row(inp, button_conversation),
        pn.pane.Markdown("## 对话历史"),
        conversation_panel,
        width=800
    )

    return dashboard

# 直接在模块顶层调用
dashboard = create_qa_interface(debug=True)
dashboard.servable()