import streamlit as st
from src.data_processor import VehicleDataLoader
from src.llm_engine import LLMEngine
import time

# --- 界面配置 ---
st.set_page_config(
    page_title="车辆维修资料助手", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 核心限制常量
MAX_AUTO_SHOW = 5
MAX_MEMORY_ROUNDS = 10 

@st.cache_resource
def load_engines():
    data_loader = VehicleDataLoader("data/raw_data.csv")
    llm_bot = LLMEngine()
    return data_loader, llm_bot
data_loader, llm_bot = load_engines()

# === 🚑 紧急诊断代码 (调试完可删除) ===
import os
st.error("正在进行环境诊断...")
# 1. 检查当前目录
st.write(f"当前工作目录: `{os.getcwd()}`")
# 2. 检查 data 文件夹是否存在
if os.path.exists("data"):
    st.write(f"📂 data 文件夹下的文件: `{os.listdir('data')}`")
else:
    st.write("❌ 警告：找不到 'data' 文件夹！")
# 3. 检查数据是否加载进内存
if data_loader.df is not None and not data_loader.df.empty:
    st.success(f"✅ 数据加载成功！共 {len(data_loader.df)} 行。")
    st.dataframe(data_loader.df.head(3)) # 展示前3行看看有没有乱码
else:
    st.error("❌ 数据加载失败！DataFrame 为空。")
# ========================================


data_loader, llm_bot = load_engines()

if "messages" not in st.session_state: st.session_state.messages = []
if "search_history" not in st.session_state: st.session_state.search_history = []

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 操作面板")
    st.markdown("---")
    if st.button("⬅️ 返回上一步 (撤销)", use_container_width=True):
        if len(st.session_state.messages) >= 2:
            st.session_state.messages.pop()
            st.session_state.messages.pop()
            if st.session_state.search_history:
                st.session_state.search_history.pop()
            st.rerun()
        else:
            st.warning("已经是第一步了")

    if st.button("🔄 重新开始对话", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.search_history = []
        st.rerun()

# --- 核心交互 ---
def handle_input(user_input, is_option_click=False):
    # 1. 显示用户提问
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        msg_placeholder.markdown("🤔 正在深度分析...")
        
        # --- A. 意图识别与关键词处理 ---
        # 获取上一轮的搜索词列表 (list)
        current_kws = st.session_state.search_history[-1] if st.session_state.search_history else []
        
        target_kws = []
        intent_dict = {} # 用于结构化打分的字典
        status = ""
        
        if is_option_click:
            # 点击选项：视为简单的追加关键词
            target_kws = current_kws + [user_input]
            intent_dict = {"all_keywords": target_kws} # 构造简单字典
            status = f"🔍 已在 **{' '.join(current_kws)}** 基础上增加：**{user_input}**"
        else:
            # 自然语言输入：调用 LLM 获取结构化意图
            intent_dict = llm_bot.extract_intent(user_input)
            extracted_list = intent_dict.get("all_keywords", [])
            
            # 尝试追加 (将新词和旧词合并)
            try_combine = list(set(current_kws + extracted_list))
            # 探测性搜索
            test_res = data_loader.search(try_combine)
            
            if not test_res.empty and len(current_kws) > 0:
                # 成功追加
                target_kws = try_combine
                # 在追加模式下，我们主要依赖关键词列表，结构化字段可能不准确，所以简化 intent_dict
                intent_dict = {"all_keywords": target_kws}
                status = f"🔍 正在筛选：**{' '.join(target_kws)}**"
            else:
                # 新搜索
                target_kws = extracted_list
                # intent_dict 保持 LLM 返回的完整结构 (包含 brand, series 等)
                status = f"🔄 新搜索：**{' '.join(target_kws)}**"

        msg_placeholder.markdown(status)

        # --- B. 检索逻辑 ---
        # 1. 尝试精准匹配 (Filter)
        all_results = data_loader.search(target_kws)
        is_fuzzy = False
        
        # 2. 如果精准无果，尝试模糊推荐 (Search Best Match)
        if all_results.empty:
            # 传入 intent_dict 以触发结构化加权
            all_results = data_loader.search_best_match(intent_dict, top_n=100)
            is_fuzzy = True

        count = len(all_results)
        options = []
        reply = ""

        if count == 0:
            reply = f"❌ **抱歉，未找到相关结果。**\n\n当前关键词：`{' + '.join(target_kws)}`"
        else:
            # 存入历史
            st.session_state.search_history.append(target_kws)
            if len(st.session_state.search_history) > MAX_MEMORY_ROUNDS:
                st.session_state.search_history.pop(0)

            # --- C. 强制展示 Top 5 ---
            # 使用结构化打分排序
            top_5_display = data_loader.search_best_match(intent_dict, top_n=MAX_AUTO_SHOW)
            
            if is_fuzzy:
                reply = f"⚠️ **未找到完全匹配的结果，为您推荐 {count} 条最相关的资料：**\n\n"
            else:
                reply = f"✅ **找到 {count} 条相关资料，其中最匹配的前 {len(top_5_display)} 条如下：**\n\n"
            
            for _, row in top_5_display.iterrows():
                reply += f"- 📄 **【ID: {row['id']}】** {row['filename']}\n"

            # --- D. 生成选项 ---
            if count > 1:
                question_text, options = data_loader.generate_dynamic_options(all_results, target_kws)
                if options:
                    reply += f"\n💡 **若以上没有您需要的，您是否想找：**"
                else:
                    if count > MAX_AUTO_SHOW:
                        reply += "\n*(结果较多且相似度高，建议手动输入更具体的型号)*"

        # 渲染文本
        msg_placeholder.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply, "options": options})
        
        # --- E. 渲染按钮 ---
        if options:
            cols = st.columns(4)
            msg_idx = len(st.session_state.messages) - 1
            for idx, opt in enumerate(options):
                if cols[idx % 4].button(str(opt), key=f"btn_{msg_idx}_{idx}"):
                    handle_input(str(opt), is_option_click=True)
                    st.rerun()

# --- 历史回放 ---
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("options") and i == len(st.session_state.messages) - 1:
            cols = st.columns(4)
            for idx, opt in enumerate(msg["options"]):
                if cols[idx % 4].button(str(opt), key=f"btn_{i}_{idx}"):
                    handle_input(str(opt), is_option_click=True)
                    st.rerun()

if prompt := st.chat_input("请输入车型、故障或零件名称..."):
    if prompt in ["重置", "清空"]:
        st.session_state.clear()
        st.rerun()
    handle_input(prompt)