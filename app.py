import streamlit as st
import pandas as pd
from src.data_processor import VehicleDataLoader
from src.llm_engine import LLMEngine
import time

# --- 界面配置 ---
st.set_page_config(
    page_title="智能车辆电路图资料导航 Chatbot", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 核心限制常量
MAX_AUTO_SHOW = 5  
MAX_MEMORY_ROUNDS = 10 

# --- 初始化核心引擎 ---
if "data_loader" not in st.session_state:
    st.session_state.data_loader = VehicleDataLoader("data/raw_data.csv")

if "llm_bot" not in st.session_state:
    st.session_state.llm_bot = LLMEngine()

# 便捷引用
data_loader = st.session_state.data_loader
llm_bot = st.session_state.llm_bot

# --- Session 初始化 ---
if "messages" not in st.session_state: st.session_state.messages = []
if "search_history" not in st.session_state: st.session_state.search_history = []

# --- 侧边栏：控制面板 ---
with st.sidebar:
    st.title("⚙️ 控制面板")
    st.markdown("---")
    
    # 功能 1: 返回上一步
    if st.button("⬅️ 返回上一步", use_container_width=True):
        if len(st.session_state.messages) >= 2:
            st.session_state.messages.pop()
            st.session_state.messages.pop()
            if st.session_state.search_history:
                st.session_state.search_history.pop()
            st.rerun()
        else:
            st.toast("已经是第一步了", icon="⚠️")

    # 功能 2: 分隔记忆 (开启新话题)
    if st.button("✨ 开启新话题 (分隔记忆)", use_container_width=True):
        if st.session_state.search_history:
            # 只清空搜索条件的记忆栈，不删聊天记录
            st.session_state.search_history = []
            # 在界面上插入一条分割线消息
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "--- ✂️ **记忆已重置，请开始新的搜索** ---",
                "options": []
            })
            st.rerun()

    # 功能 3: 清除记录 (重置)
    if st.button("🗑️ 清除聊天记录", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.search_history = []
        st.rerun()

    # 管理员功能 (保留)
    st.markdown("---")
    with st.expander("🛠️ 数据管理 (管理员)"):
        st.caption("当前数据总数: " + str(len(data_loader.df)))
        
        if st.button("⚡ 重置为初始数据"):
            st.cache_resource.clear()
            st.session_state.data_loader = VehicleDataLoader("data/raw_data.csv")
            if "last_processed_file" in st.session_state:
                del st.session_state.last_processed_file
            st.toast("数据已重置为初始状态！", icon="✅")
            time.sleep(1)
            st.rerun()
            
        uploaded_file = st.file_uploader("追加新CSV数据", type=["csv"], key="csv_uploader")
        
        if uploaded_file is not None:
            current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if "last_processed_file" not in st.session_state or st.session_state.last_processed_file != current_file_id:
                try:
                    try:
                        raw_new_df = pd.read_csv(uploaded_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        raw_new_df = pd.read_csv(uploaded_file, encoding='gbk')
                    
                    temp_loader = VehicleDataLoader(dataframe=raw_new_df)
                    clean_new_df = temp_loader.df
                    
                    if clean_new_df is not None and not clean_new_df.empty:
                        old_df = st.session_state.data_loader.df
                        if old_df is not None and not old_df.empty:
                            merged_df = pd.concat([old_df, clean_new_df], ignore_index=True)
                            if 'id' in merged_df.columns:
                                merged_df.drop_duplicates(subset=['id'], keep='last', inplace=True)
                        else:
                            merged_df = clean_new_df

                        st.session_state.data_loader = VehicleDataLoader(dataframe=merged_df)
                        st.session_state.last_processed_file = current_file_id
                        
                        st.toast(f"🎉 成功追加 {len(clean_new_df)} 条数据，当前总数: {len(merged_df)}", icon="✅")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("上传的文件格式不正确或为空")
                        
                except Exception as e:
                    st.error(f"数据加载失败: {e}")

# --- 核心交互 ---
def handle_input(user_input, is_option_click=False):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        msg_placeholder.markdown("🤔 正在检索资料库...")
        
        # --- A. 关键词处理 ---
        current_kws = st.session_state.search_history[-1] if st.session_state.search_history else []
        target_kws = []
        status = ""
        
        if is_option_click:
            target_kws = current_kws + [user_input]
            status = f"🔍 已在 **{' '.join(current_kws)}** 基础上增加：**{user_input}**"
        else:
            extracted = llm_bot.extract_intent(user_input)
            extracted_list = extracted.get("all_keywords", [])
            
            try_combine = list(set(current_kws + extracted_list))
            test_res = data_loader.search(try_combine)
            
            if not test_res.empty and len(current_kws) > 0:
                target_kws = try_combine
                extracted["all_keywords"] = target_kws
                status = f"🔍 正在筛选：**{' '.join(target_kws)}**"
            else:
                target_kws = extracted_list
                status = f"🔄 新搜索：**{' '.join(target_kws)}**"
            
            current_intent = extracted

        msg_placeholder.markdown(status)

        # --- B. 检索逻辑 ---
        all_results = data_loader.search(target_kws)
        is_fuzzy = False
        
        if is_option_click:
            current_intent = {"all_keywords": target_kws}

        if all_results.empty:
            all_results = data_loader.search_best_match(current_intent, top_n=100)
            is_fuzzy = True

        count = len(all_results)
        options = []
        reply = ""

        if count == 0:
            reply = f"❌ **抱歉，未找到相关结果。**\n\n当前关键词：`{' + '.join(target_kws)}`"
        else:
            st.session_state.search_history.append(target_kws)
            if len(st.session_state.search_history) > MAX_MEMORY_ROUNDS:
                st.session_state.search_history.pop(0)

            # --- C. 展示 Top 5 ---
            top_5_display = data_loader.search_best_match(current_intent, top_n=MAX_AUTO_SHOW)
            
            if is_fuzzy:
                reply = f"⚠️ **未找到完全匹配的结果，为您推荐 {count} 条最相关的资料：**\n\n"
            else:
                reply = f"✅ **找到 {count} 条相关资料，其中最匹配的前 {len(top_5_display)} 条如下：**\n\n"
            
            for _, row in top_5_display.iterrows():
                reply += f"- 📄 **【ID: {row['id']}】** {row['filename']}\n"

            # --- D. 生成选项 (仅当结果数量 > 5 时才生成) ---
            if count > MAX_AUTO_SHOW:
                question_text, options = data_loader.generate_dynamic_options(all_results, target_kws)
                if options:
                    reply += f"\n💡 **若以上没有您需要的，您是否想找：**"
                else:
                    reply += "\n*(结果较多且相似度高，建议手动输入更具体的型号)*"

        # 渲染
        msg_placeholder.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply, "options": options})
        
        # 按钮
        if options:
            cols = st.columns(4)
            msg_idx = len(st.session_state.messages) - 1
            for idx, opt in enumerate(options):
                if cols[idx % 4].button(str(opt), key=f"btn_{msg_idx}_{idx}"):
                    handle_input(str(opt), is_option_click=True)
                    st.rerun()

# --- 历史回放 ---
st.title("🚗 智能车辆维修资料导航")

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 仅在最后一条消息显示按钮
        if msg.get("options") and i == len(st.session_state.messages) - 1:
            cols = st.columns(4)
            for idx, opt in enumerate(msg["options"]):
                if cols[idx % 4].button(str(opt), key=f"btn_{i}_{idx}"):
                    handle_input(str(opt), is_option_click=True)
                    st.rerun()

# --- 底部输入 ---
if prompt := st.chat_input("请输入车型、故障或零件名称..."):
    # 为了兼容习惯，如果用户在输入框打'重置'也触发清空
    if prompt in ["重置", "清空"]:
        st.session_state.messages = []
        st.session_state.search_history = []
        st.rerun()
    handle_input(prompt)