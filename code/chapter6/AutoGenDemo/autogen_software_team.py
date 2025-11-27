"""
AutoGen 软件开发团队协作案例
"""

import os
import asyncio
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 先测试一个版本，使用 OpenAI 客户端
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

def create_openai_model_client():
    """创建 OpenAI 模型客户端用于测试"""

    # 增加modelinfo
    model_info_env = os.getenv("LLM_MODEL_INFO")
    model_info = None
    if model_info_env:
        try:
            model_info = json.loads(model_info_env)
        except Exception:
            model_info = None
    return OpenAIChatCompletionClient(
        model=os.getenv("LLM_MODEL_ID", "gpt-4o"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        model_info=model_info,
    )

def create_product_manager(model_client):
    """创建产品经理智能体"""
    system_message = """你是一位经验丰富的产品经理，专门负责软件产品的需求分析和项目规划。

你的核心职责包括：
1. **需求分析**：深入理解用户需求，识别核心功能和边界条件
2. **技术规划**：基于需求制定清晰的技术实现路径
3. **风险评估**：识别潜在的技术风险和用户体验问题
4. **协调沟通**：与工程师和其他团队成员进行有效沟通

当接到开发任务时，请按以下结构进行分析：
1. 需求理解与分析
2. 功能模块划分
3. 技术选型建议
4. 实现优先级排序
5. 验收标准定义

请简洁明了地回应，并在分析完成后说"请工程师开始实现"。"""

# 上层 Agent/Team 自动构建 messages （含系统与用户），客户端只是发送。
    return AssistantAgent(
        name="ProductManager",
        model_client=model_client,
        system_message=system_message,
    )

def create_engineer(model_client):
    """创建软件工程师智能体"""
    system_message = """你是一位资深的软件工程师，擅长 Python 开发和 Web 应用构建。

你的技术专长包括：
1. **Python 编程**：熟练掌握 Python 语法和最佳实践
2. **Web 开发**：精通 Streamlit、Flask、Django 等框架
3. **API 集成**：有丰富的第三方 API 集成经验
4. **错误处理**：注重代码的健壮性和异常处理

当收到开发任务时，请：
1. 仔细分析技术需求
2. 选择合适的技术方案
3. 编写完整的代码实现
4. 添加必要的注释和说明
5. 考虑边界情况和异常处理

请提供完整的可运行代码，并在完成后说"请代码审查员检查"。
如果在实现过程中发现需求不明确、存在冲突或需要产品经理变更，请输出"ROLLBACK_TO_PM"并简要说明原因和建议变更点。"""

    return AssistantAgent(
        name="Engineer",
        model_client=model_client,
        system_message=system_message,
    )

def create_code_reviewer(model_client):
    """创建代码审查员智能体"""
    system_message = """你是一位经验丰富的代码审查专家，专注于代码质量和最佳实践。

你的审查重点包括：
1. **代码质量**：检查代码的可读性、可维护性和性能
2. **安全性**：识别潜在的安全漏洞和风险点
3. **最佳实践**：确保代码遵循行业标准和最佳实践
4. **错误处理**：验证异常处理的完整性和合理性

审查流程：
1. 仔细阅读和理解代码逻辑
2. 检查代码规范和最佳实践
3. 识别潜在问题和改进点
4. 提供具体的修改建议
5. 评估代码的整体质量

请提供具体的审查意见，完成后说"代码审查完成，请用户代理测试"。
如果发现需要从需求层面调整，请输出"ROLLBACK_TO_PM"并说明理由与建议。"""

    return AssistantAgent(
        name="CodeReviewer",
        model_client=model_client,
        system_message=system_message,
    )

def create_user_proxy():
    """创建用户代理智能体"""
    return UserProxyAgent(
        name="UserProxy",
        description="""用户代理，负责以下职责：
1. 代表用户提出开发需求
2. 执行最终的代码实现
3. 验证功能是否符合预期
4. 提供用户反馈和建议

完成测试后请回复 TERMINATE。""",
    )

async def run_software_development_team():
    """运行软件开发团队协作"""
    
    print("🔧 正在初始化模型客户端...")
    # 先使用标准的OpenAI客户端测试
    model_client = create_openai_model_client()
    
    print("👥 正在创建智能体团队...")
    
    # 创建智能体团队
    product_manager = create_product_manager(model_client)
    engineer = create_engineer(model_client)
    code_reviewer = create_code_reviewer(model_client)
    user_proxy = create_user_proxy()
    
    task = """我们需要开发一个比特币价格显示应用，具体要求如下：

核心功能：
- 实时显示比特币当前价格（USD）
- 显示24小时价格变化趋势（涨跌幅和涨跌额）
- 提供价格刷新功能

技术要求：
- 使用 Streamlit 框架创建 Web 应用
- 界面简洁美观，用户友好
- 添加适当的错误处理和加载状态

请团队协作完成这个任务，从需求分析到最终实现。"""
    
    # 执行团队协作
    print("🚀 启动动态协作工作流...")
    print("=" * 60)
    
    current_task = task
    final_result = None

    while True:
        pm_chat = RoundRobinGroupChat(
            participants=[product_manager],
            termination_condition=TextMentionTermination("请工程师开始实现"),
            max_turns=5,
        )
        pm_result = await Console(pm_chat.run_stream(task=current_task))

        eng_chat = RoundRobinGroupChat(
            participants=[engineer],
            termination_condition=TextMentionTermination("请代码审查员检查"),
            max_turns=10,
        )
        eng_result = await Console(eng_chat.run_stream(task=pm_result))
        if isinstance(eng_result, str) and "ROLLBACK_TO_PM" in eng_result:
            current_task = f"产品经理请根据工程师的回退请求重新评估并更新需求：\n{eng_result}"
            continue

        review_chat = RoundRobinGroupChat(
            participants=[code_reviewer],
            termination_condition=TextMentionTermination("代码审查完成，请用户代理测试"),
            max_turns=5,
        )
        review_result = await Console(review_chat.run_stream(task=eng_result))
        if isinstance(review_result, str) and "ROLLBACK_TO_PM" in review_result:
            current_task = f"产品经理请根据代码审查意见重新评估并更新需求：\n{review_result}"
            continue

        user_chat = RoundRobinGroupChat(
            participants=[user_proxy],
            termination_condition=TextMentionTermination("TERMINATE"),
            max_turns=5,
        )
        final_result = await Console(user_chat.run_stream(task=review_result))
        break
    
    print("\n" + "=" * 60)
    print("✅ 团队协作完成！")
    
    return final_result

# 主程序入口
if __name__ == "__main__":
    try:
        # 运行异步协作流程
        result = asyncio.run(run_software_development_team())
        
        print(f"\n📋 协作结果摘要：")
        print(f"- 参与智能体数量：4个")
        print(f"- 任务完成状态：{'成功' if result else '需要进一步处理'}")
        
    except ValueError as e:
        print(f"❌ 配置错误：{e}")
        print("请检查 .env 文件中的配置是否正确")
    except Exception as e:
        print(f"❌ 运行错误：{e}")
        import traceback
        traceback.print_exc()



