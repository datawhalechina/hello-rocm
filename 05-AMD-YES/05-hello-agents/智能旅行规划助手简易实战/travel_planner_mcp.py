"""
AMD395 x HelloAgents x MCP hands-on example: intelligent travel planning assistant.

Environment requirements:
1. Start the Qwen3-30B model on the AMD395 machine (port 1234 by default; adjust if your port differs).
2. Install HelloAgents: pip install hello-agents==0.2.8
3. Install uv: pip install uv
4. Apply for an AMap API key: https://lbs.amap.com/

Run:
python travel_planner_mcp.py

AMD395 × HelloAgents × MCP实战：智能旅行规划助手


环境要求：
1. AMD395机器已启动Qwen3-30B模型（端口1234）(根据使用的端口可能需要调整)
2. 已安装HelloAgents：pip install hello-agents==0.2.8
3. 已安装uv工具：pip install uv
4. 申请高德地图API Key：https://lbs.amap.com/

运行方式：
python travel_planner_mcp.py
"""

import os
import logging
from hello_agents import SimpleAgent
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.tools import MCPTool

# Configure logging to show only WARNING level and above.
# 配置日志 - 只显示WARNING级别以上
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logging.getLogger('openai').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)


class MultiAgentTravelPlanner:
    """Multi-agent travel planning system.

    多Agent旅行规划系统
    """

    def __init__(self, amap_api_key: str):
        """Initialize the multi-agent system.

        初始化多Agent系统
        """
        print("\n🔄 初始化多Agent旅行规划系统...")

        # 1. Connect to the local AMD395-hosted model.
        # 1. 连接AMD395本地模型
        print("  ├─ 连接AMD395...")
        self.llm = HelloAgentsLLM(
            model="Qwen3-30B-2507-instruct",
            base_url="http://127.0.0.1:1234/v1", # Adjust this if your local endpoint uses a different port.
            # (根据使用的端口可能需要调整)
            api_key="amd395"
        )

        # Test the model connection.
        # 测试连接
        try:
            messages = [{"role": "user", "content": "你好"}]
            response = ""
            for chunk in self.llm.think(messages):
                response += chunk
            print("  ├─ ✓ 模型连接成功")
        except Exception as e:
            print(f"  └─ ✗ 模型连接失败：{e}")
            raise

        # 2. Create one shared MCP tool instance for all agents.
        # 2. 创建共享的MCP工具（关键：只创建一次，多个Agent共享）
        print("  ├─ 创建共享的高德地图MCP工具...")
        self.amap_tool = MCPTool(
            name="amap",
            description="高德地图服务",
            server_command=["uvx", "amap-mcp-server"],
            env={"AMAP_MAPS_API_KEY": amap_api_key},
            auto_expand=True
        )
        print("  ├─ ✓ MCP工具创建成功")

        # 3. Create the attraction-search agent.
        # 3. 创建景点搜索Agent（只负责搜索景点）
        print("  ├─ 创建景点搜索Agent...")
        self.attraction_agent = SimpleAgent(
            name="景点搜索专家",
            llm=self.llm,
            system_prompt="""你是景点搜索专家。

你的任务：
使用amap_maps_text_search工具搜索指定城市的景点。

输出格式：
请按JSON格式返回景点列表，每个景点包含：
- name: 景点名称
- address: 详细地址
- type: 景点类型

示例：
[
  {"name": "西湖风景区", "address": "浙江省杭州市西湖区", "type": "自然风光"},
  {"name": "灵隐寺", "address": "浙江省杭州市西湖区灵隐路", "type": "历史文化"}
]
"""
        )
        self.attraction_agent.add_tool(self.amap_tool)

        # 4. Create the weather-query agent.
        # 4. 创建天气查询Agent（只负责查询天气）
        print("  ├─ 创建天气查询Agent...")
        self.weather_agent = SimpleAgent(
            name="天气查询专家",
            llm=self.llm,
            system_prompt="""你是天气查询专家。

你的任务：
使用amap_maps_weather工具查询指定城市的天气情况。

输出格式：
请按JSON格式返回天气信息，包含：
- city: 城市名称
- weather: 天气状况
- temperature: 温度
- wind: 风力风向

示例：
{
  "city": "杭州",
  "weather": "晴",
  "temperature": "10-18℃",
  "wind": "东风1-3级"
}
"""
        )
        self.weather_agent.add_tool(self.amap_tool)

        # 5. Create the itinerary-planning agent, which only synthesizes information.
        # 5. 创建行程规划Agent（不调用工具，只负责整合信息）
        print("  ├─ 创建行程规划Agent...")
        self.planner_agent = SimpleAgent(
            name="行程规划专家",
            llm=self.llm,
            system_prompt="""你是行程规划专家。

你的任务：
根据景点信息和天气信息，生成详细的旅行规划。

输出格式（Markdown）：
# {城市}N日游旅行规划

## 行程概览
- 目的地：...
- 旅行天数：...
- 总预算：...

## 天气情况
（根据天气信息填写）

## 详细行程
### 第1天：...
- 上午：...
- 下午：...
- 晚上：...
- 预算：...元

### 第2天：...
...

## 预算总结
...

## 旅行建议
...
"""
        )

        print(f"✅ 多Agent系统初始化成功")
        print(f"   景点搜索Agent: {len(self.attraction_agent.list_tools())} 个工具")
        print(f"   天气查询Agent: {len(self.weather_agent.list_tools())} 个工具")
        print(f"   行程规划Agent: {len(self.planner_agent.list_tools())} 个工具\n")

    def plan_travel(self, destination: str, days: int, budget: float, preferences: str = ""):
        """
        Plan a trip through multi-agent collaboration.

        Workflow:
        1. Attraction agent -> search attractions
        2. Weather agent -> query weather
        3. Planner agent -> merge results into a final itinerary

        使用多Agent协作规划旅行

        流程：
        1. 景点搜索Agent → 搜索景点
        2. 天气查询Agent → 查询天气
        3. 行程规划Agent → 整合信息生成计划
        """
        print("=" * 80)
        print("【开始多Agent协作规划旅行】")
        print("=" * 80)
        print(f"目的地：{destination}")
        print(f"天数：{days}天")
        print(f"预算：{budget}元")
        print(f"偏好：{preferences if preferences else '不限'}")
        print("=" * 80)
        print()

        # Step 1: use the attraction agent to search for attractions.
        # 步骤1: 景点搜索Agent搜索景点
        print("📍 步骤1: 景点搜索Agent工作中...")
        if preferences:
            keywords = preferences  # Use the user's preferences as the search keywords.
            # 使用用户偏好作为关键词
        else:
            keywords = "景点"

        attraction_query = f"请使用amap_maps_text_search工具搜索{destination}的{keywords}相关景点，返回前10个结果"
        print(f"   查询: {attraction_query}")
        attraction_response = self.attraction_agent.run(attraction_query)
        print(f"   ✓ 景点搜索完成\n")

        # Step 2: use the weather agent to fetch weather information.
        # 步骤2: 天气查询Agent查询天气
        print("🌤️  步骤2: 天气查询Agent工作中...")
        weather_query = f"请使用amap_maps_weather工具查询{destination}的天气信息"
        print(f"   查询: {weather_query}")
        weather_response = self.weather_agent.run(weather_query)
        print(f"   ✓ 天气查询完成\n")

        # Step 3: let the planner agent synthesize the final plan.
        # 步骤3: 行程规划Agent整合信息生成计划
        print("📋 步骤3: 行程规划Agent整合信息中...")
        print("   (这一步可能需要30-60秒，请耐心等待...)")

        # Keep the planner prompt compact to avoid overlong context.
        # 简化传递给规划Agent的信息（避免Prompt过长）
        planner_query = f"""请为我生成一个{destination}{days}日游的详细旅行规划：

【基本信息】
- 目的地：{destination}
- 旅行天数：{days}天
- 总预算：{budget}元
- 偏好：{preferences if preferences else "不限"}

【天气情况】
{weather_response[:500]}...

【可选景点】
{attraction_response[:800]}...

【要求】
1. 生成{days}天的详细行程安排
2. 每天安排2-3个主要景点
3. 包含住宿、餐饮、交通建议
4. 预算分配合理
5. 以Markdown格式输出

请开始生成规划："""

        print(f"   正在生成规划（约需30秒）...")

        # Run the planner agent. This step can take longer.
        # 调用规划Agent（这一步可能比较慢）
        result = self.planner_agent.run(planner_query)

        print(f"   ✓ 行程规划完成\n")

        return result





def save_to_markdown(content: str, destination: str, days: int):
    """Save the generated itinerary to a Markdown file.

    保存行程到Markdown文件
    """
    filename = f"{destination}_{days}日游_MCP.md"

    with open(filename, "w", encoding="utf-8") as f:
        # Add only the signature because the agent output already contains a title.
        # 只添加署名，不添加标题（Agent生成的内容已经包含标题）
        f.write(f"*本行程由AMD395×HelloAgents×MCP自动生成*\n\n")
        f.write("---\n\n")
        f.write(content)

    print(f"\n✓ 行程已保存到文件：{filename}")
    return filename


def main():
    """Main entry point.

    主函数
    """
    print("=" * 80)
    print("AMD395 × HelloAgents × MCP实战：智能旅行规划助手")
    print("=" * 80)

    # Configure the AMap API key.
    # 配置高德地图API Key
    amap_api_key = "your_api_key_here"  # Replace this with your own API key.
    # 替换成你自己的API Key

    # Or load it from an environment variable.
    # 或从环境变量读取
    if not amap_api_key:
        amap_api_key = os.getenv("AMAP_MAPS_API_KEY")

    if not amap_api_key:
        print("\n✗ 错误：未配置高德地图API Key")
        print("\n请修改脚本第261行，填写你的API Key：")
        print('  amap_api_key = "your_api_key_here"')
        print("\n申请地址：https://lbs.amap.com/")
        return

    print(f"\n✓ 已配置高德地图API Key: {amap_api_key[:8]}...{amap_api_key[-4:]}")

    try:
        # Create the multi-agent system; the MCP tool stays alive here.
        # 创建多Agent旅行规划系统（MCP工具将在这里创建并保持活跃）
        planner = MultiAgentTravelPlanner(amap_api_key)

        # Plan the trip.
        # 规划旅行
        print("=" * 80)
        print("【开始旅行规划】")
        print("=" * 80)
        result = planner.plan_travel(
            destination="杭州",
            days=3,
            budget=3000,
            preferences="自然风光和历史文化"
        )

        # Print the generated result.
        # 输出结果
        print("\n" + "=" * 80)
        print("【规划完成】")
        print("=" * 80)
        print(result)

        # Save the itinerary to disk.
        # 保存到文件
        save_to_markdown(result, "杭州", 3)

        print("\n" + "=" * 80)
        print("✓ 旅行规划完成！")
        print("=" * 80)
        print("\n【私有化部署优势】")
        print("✓ 数据隐私：旅行偏好、个人信息等敏感数据全程本地处理，不上传云端")
        print("✓ 成本可控：无需支付云端API费用，可以无限次调用优化行程")
        print("✓ 响应迅速：局域网直连，延迟低于云端API")
        print("✓ 自主可控：可以随时调整模型参数和系统提示词")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ 发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
