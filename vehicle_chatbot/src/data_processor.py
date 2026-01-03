import pandas as pd
import re
from collections import Counter
import math  # 引入 math 做对数运算

class VehicleDataLoader:
    def __init__(self, csv_path="data/raw_data.csv"):
        self.csv_path = csv_path
        self.df = None
        self.load_data()

    def load_data(self):
        try:
            try:
                self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                self.df = pd.read_csv(self.csv_path, encoding='gbk')

            new_columns = {}
            for col in self.df.columns:
                if "文件名称" in col or "关联文件" in col:
                    new_columns[col] = "filename"
                elif "层级" in col:
                    new_columns[col] = "hierarchy"
                elif "ID" in col:
                    new_columns[col] = "id"
            self.df.rename(columns=new_columns, inplace=True)
            self.df.fillna("", inplace=True)

            def parse_hierarchy(val):
                txt = str(val).replace('->', '>').replace('—>', '>')
                return [t.strip() for t in txt.split('>') if t.strip()]

            self.df['hierarchy_list'] = self.df['hierarchy'].apply(parse_hierarchy)
            self.df['full_text'] = self.df['hierarchy'].astype(str) + " " + self.df['filename'].astype(str)
            return self.df
        except Exception as e:
            return pd.DataFrame()

    def search(self, keywords):
        """精准搜索 (保持不变，依然用列表搜)"""
        if self.df is None or self.df.empty: return pd.DataFrame()
        if not keywords: return self.df
        mask = pd.Series([True] * len(self.df))
        for kw in keywords:
            kw = kw.strip()
            if not kw: continue
            condition = self.df['full_text'].str.contains(re.escape(kw), case=False, na=False)
            mask = mask & condition
        return self.df[mask]

    def search_best_match(self, intent_dict, top_n=5):
        """
        [核心算法 V8.0] 结构化加权 + 平滑IDF
        Input: intent_dict = {'brand': '红岩', 'all_keywords': [...]}
        """
        if self.df is None or self.df.empty: return pd.DataFrame()
        
        # 兼容旧逻辑：如果传进来的是 list，转成 dict
        if isinstance(intent_dict, list):
            keywords = intent_dict
            intent_dict = {"all_keywords": keywords}
        else:
            keywords = intent_dict.get("all_keywords", [])

        if not keywords: return pd.DataFrame()

        # 1. 计算平滑 IDF
        keyword_idf = {}
        total_docs = len(self.df)
        for kw in keywords:
            kw = kw.strip()
            if not kw: continue
            doc_freq = self.df['full_text'].str.contains(re.escape(kw), case=False, na=False).sum()
            # Log 平滑：log(N / (n+1)) + 1
            # 这样 1次出现的词权重约为 log(4000)≈8，几百次出现的词约为 log(10)≈2
            # 差距从 100倍 缩小到了 4倍，更合理
            weight = math.log((total_docs + 1) / (doc_freq + 1)) + 1.0
            keyword_idf[kw] = weight

        # 2. 结构化打分函数
        def calculate_score(row):
            score = 0.0
            full_str = str(row['full_text']).lower()
            
            # A. 基础分：基于所有关键词的 IDF
            for kw, weight in keyword_idf.items():
                if kw.lower() in full_str:
                    score += weight

            # B. 结构化加分 (Semantic Boost)
            # 如果 LLM 识别出这是"品牌"，且该词出现在了层级的前部 -> 巨额加分
            if intent_dict.get("brand"):
                brand = intent_dict["brand"].lower()
                # 检查层级列表的前3层是否有这个品牌
                h_list = row['hierarchy_list']
                for i in range(min(3, len(h_list))):
                    if brand in h_list[i].lower():
                        score += 5.0 # 品牌匹配非常重要，直接加分
                        break
            
            # 如果 LLM 识别出"部件"，且出现在文件名里 -> 巨额加分
            if intent_dict.get("component"):
                comp = intent_dict["component"].lower()
                if comp in str(row['filename']).lower():
                    score += 8.0 # 部件匹配最核心
            
            return score

        # 3. 计算与排序
        # 同样先筛选候选集
        candidate_mask = pd.Series([False] * len(self.df))
        for kw in keywords:
            candidate_mask |= self.df['full_text'].str.contains(re.escape(kw), case=False, na=False)
        
        candidate_df = self.df[candidate_mask].copy()
        if candidate_df.empty: return pd.DataFrame()

        candidate_df['match_score'] = candidate_df.apply(calculate_score, axis=1)
        return candidate_df.sort_values(by='match_score', ascending=False).head(top_n)

    # ... generate_dynamic_options 和 _generate_filename_options 保持 V5.0 原样 ...
    # 为节省篇幅，请保留你之前文件里的这两个函数，不要删掉！
    # 也就是把上面的 search_best_match 替换掉原来的，其他不动。
    def _generate_filename_options(self, current_df):
        # (保持原样)
        all_tokens = []
        stop_words = {'电路图', '维修', '手册', '定义', '针脚', '保险', '原理图', '整车', '上汽', '红岩', '杰狮', '东风', '解放'}
        for name in current_df['filename']:
            tokens = re.split(r'[^a-zA-Z0-9\u4e00-\u9fa5]+', str(name))
            for t in tokens:
                if len(t) > 1 and t not in stop_words:
                    all_tokens.append(t)
        counter = Counter(all_tokens)
        valid_options = []
        for word, count in counter.most_common(10):
            if 1 < count < len(current_df):
                valid_options.append(word)
        return valid_options[:5]

    def generate_dynamic_options(self, current_df, user_keywords=[]):
        # (保持原样)
        if current_df is None or current_df.empty: return None, []
        max_depth = 12
        start_check_level = 0
        if user_keywords:
            sample_rows = current_df.head(50) 
            for _, row in sample_rows.iterrows():
                h_list = row['hierarchy_list']
                for kw in user_keywords:
                    for idx, tag in enumerate(h_list):
                        if kw.lower() in tag.lower() or tag.lower() in kw.lower():
                            if idx + 1 > start_check_level:
                                start_check_level = idx + 1
        target_level = -1
        for i in range(max_depth - 1, start_check_level - 1, -1):
            vals = set()
            for _, row in current_df.iterrows():
                if i < len(row['hierarchy_list']):
                    val = row['hierarchy_list'][i]
                    is_existing = False
                    for kw in user_keywords:
                        if kw in val or val in kw:
                            is_existing = True
                            break
                    if not is_existing:
                        vals.add(val)
            if 1 < len(vals) <= 20:
                target_level = i
                break
        if target_level != -1:
            all_vals = []
            for _, row in current_df.iterrows():
                if target_level < len(row['hierarchy_list']):
                    val = row['hierarchy_list'][target_level]
                    is_existing = False
                    for kw in user_keywords:
                        if kw in val or val in kw:
                            is_existing = True
                            break
                    if not is_existing:
                        all_vals.append(val)
            counter = Counter(all_vals)
            sorted_options = [k for k, v in counter.most_common()]
            if sorted_options:
                return "请选择更具体的分类：", sorted_options
        filename_options = self._generate_filename_options(current_df)
        final_opts = []
        for opt in filename_options:
            is_existing = False
            for kw in user_keywords:
                if kw in opt or opt in kw:
                    is_existing = True
                    break
            if not is_existing:
                final_opts.append(opt)
        if final_opts:
            return "请选择具体的型号或配置：", final_opts
        return None, []