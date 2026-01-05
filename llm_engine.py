import streamlit as st
from openai import OpenAI
import json

class LLMEngine:
    def __init__(self):
        try:
            self.api_key = st.secrets["DEEPSEEK_API_KEY"]
            self.base_url = st.secrets["DEEPSEEK_BASE_URL"]
        except FileNotFoundError:
            self.api_key = "sk-placeholder"
            self.base_url = "https://api.deepseek.com"

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # 【关键修改】方法名变成了 extract_intent，且返回字典
    def extract_intent(self, user_query):
        """
        [V3.0] 结构化意图识别
        返回带有语义标签的字典，用于精准加权
        """
        system_prompt = """
        你是一个车辆维修资料专家。请分析用户的搜索意图，提取关键词并分类。
        
        【提取字段定义】
        - brand: 品牌 (如: 东风, 解放, 红岩, 陕汽)
        - series: 车系/车型 (如: 天龙, J6, 杰狮, M500, 欧曼)
        - component: 部件/系统 (如: 发动机, 变速箱, 仪表, ECU, 接线盒)
        - doc_type: 文档类型 (如: 电路图, 维修手册, 针脚定义)
        - other: 其他修饰词 (如: 国六, 天然气, VGT)
        
        【标准化规则】
        - 马达 -> 起动机
        - 电脑/大脑 -> ECU
        - 针角 -> 针脚
        
        【输出要求】
        - 必须输出合法的 JSON 对象。
        - 如果某字段未提及，请设为 null 或空字符串。
        - 将所有提取到的词同时也放入一个 'all_keywords' 列表中作为兜底。
        
        示例：
        用户："找一下红岩杰狮M500的接线盒图纸"
        输出：
        {
            "brand": "红岩",
            "series": "杰狮 M500",
            "component": "接线盒",
            "doc_type": "电路图",
            "other": "",
            "all_keywords": ["红岩", "杰狮", "M500", "接线盒"]
        }
        """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1,
                stream=False
            )
            content = response.choices[0].message.content.strip()
            # 清洗 Markdown 标记
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)

        except Exception as e:
            # 降级：如果解析失败，构造一个基础字典返回
            return {"all_keywords": user_query.split()}