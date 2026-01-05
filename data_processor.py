import pandas as pd
import re
from collections import Counter
import math
import numpy as np

class VehicleDataLoader:
    def __init__(self, csv_path="data/raw_data.csv", dataframe=None):
        self.csv_path = csv_path
        self.df = None
        
        if dataframe is not None:
            self.df = self._preprocess_df(dataframe)
        else:
            self.load_data()

    def _preprocess_df(self, df):
        try:
            new_columns = {}
            for col in df.columns:
                if "文件名称" in col or "关联文件" in col:
                    new_columns[col] = "filename"
                elif "层级" in col:
                    new_columns[col] = "hierarchy"
                elif "ID" in col:
                    new_columns[col] = "id"
            df.rename(columns=new_columns, inplace=True)
            df.fillna("", inplace=True)

            # 预计算全文本
            df['full_text'] = df['hierarchy'].astype(str) + " " + df['filename'].astype(str)

            def parse_hierarchy(val):
                txt = str(val).replace('->', '>').replace('—>', '>')
                return [t.strip() for t in txt.split('>') if t.strip()]

            df['hierarchy_list'] = df['hierarchy'].apply(parse_hierarchy)
            return df
        except Exception as e:
            print(f"预处理失败: {e}")
            return pd.DataFrame()

    def load_data(self):
        try:
            try:
                df = pd.read_csv(self.csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(self.csv_path, encoding='gbk')
            
            self.df = self._preprocess_df(df)
            return self.df
        except Exception as e:
            return pd.DataFrame()

    def search(self, keywords):
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
        if self.df is None or self.df.empty: return pd.DataFrame()
        
        if isinstance(intent_dict, list):
            keywords = intent_dict
            intent_dict = {"all_keywords": keywords}
        else:
            keywords = intent_dict.get("all_keywords", [])

        if not keywords: return pd.DataFrame()

        # 1. 计算 IDF (修复版)
        keyword_idf = {}
        total_docs = len(self.df)
        
        # 【修复点】确保转换为字符串类型，处理 NaN
        full_texts_np = self.df['full_text'].fillna("").astype(str).str.lower().values
        
        for kw in keywords:
            kw = kw.strip()
            if not kw: continue
            kw_lower = kw.lower()
            
            # 使用 numpy 的 char 模块进行计数
            try:
                doc_freq = np.char.count(full_texts_np, kw_lower).astype(bool).sum()
            except Exception:
                # 兜底：如果 numpy 还是报错，回退到 pandas 计算
                doc_freq = self.df['full_text'].str.contains(re.escape(kw), case=False, na=False).sum()

            weight = math.log((total_docs + 1) / (doc_freq + 1)) + 1.0
            keyword_idf[kw] = weight

        # 2. 计算得分
        def calculate_score(row):
            score = 0.0
            full_str = str(row['full_text']).lower()
            h_list = [h.lower() for h in row['hierarchy_list']]
            fname = str(row['filename']).lower()
            
            for kw, weight in keyword_idf.items():
                kw_lower = kw.lower()
                if kw_lower in full_str:
                    score += weight

            # 结构化加权
            if intent_dict.get("brand"):
                brand = intent_dict["brand"].lower()
                for i in range(min(3, len(h_list))):
                    if brand in h_list[i]:
                        score += 5.0
                        break
            
            if intent_dict.get("component"):
                comp = intent_dict["component"].lower()
                if comp in fname:
                    score += 8.0
            
            return score

        # 3. 筛选候选集 (修复版)
        candidate_mask = np.zeros(len(self.df), dtype=bool)
        for kw in keywords:
            # 确保使用 values 避免索引问题
            kw_mask = self.df['full_text'].str.contains(re.escape(kw), case=False, na=False).values
            candidate_mask |= kw_mask
            
        candidate_df = self.df[candidate_mask].copy()
        
        if candidate_df.empty: return pd.DataFrame()

        candidate_df['match_score'] = candidate_df.apply(calculate_score, axis=1)
        return candidate_df.sort_values(by='match_score', ascending=False).head(top_n)

    def generate_dynamic_options(self, current_df, user_keywords=[]):
        """[选项生成算法 - 回归版]"""
        if current_df is None or current_df.empty: return None, []

        max_depth = 12
        start_check_level = 0

        if user_keywords:
            sample_rows = current_df.head(50)
            max_hit_level = -1
            for _, row in sample_rows.iterrows():
                h_list = row['hierarchy_list']
                for kw in user_keywords:
                    for idx, tag in enumerate(h_list):
                        if kw.lower() in tag.lower() or tag.lower() in kw.lower():
                            if idx > max_hit_level:
                                max_hit_level = idx
            start_check_level = max_hit_level + 1

        target_level = -1
        
        for i in range(start_check_level, max_depth):
            vals = set()
            for _, row in current_df.iterrows():
                if i < len(row['hierarchy_list']):
                    val = row['hierarchy_list'][i]
                    is_redundant = False
                    for kw in user_keywords:
                        if kw.lower() == val.lower() or (len(kw) > 1 and kw in val):
                            is_redundant = True
                            break
                    if not is_redundant:
                        vals.add(val)

            if 1 < len(vals) <= 20:
                target_level = i
                break
        
        if target_level != -1:
            all_vals = []
            for _, row in current_df.iterrows():
                if target_level < len(row['hierarchy_list']):
                    val = row['hierarchy_list'][target_level]
                    is_redundant = False
                    for kw in user_keywords:
                        if kw.lower() == val.lower() or (len(kw) > 1 and kw in val):
                            is_redundant = True
                            break
                    if not is_redundant:
                        all_vals.append(val)
            
            counter = Counter(all_vals)
            sorted_options = [k for k, v in counter.most_common()]
            return "请选择更具体的分类：", sorted_options

        return self._generate_filename_options_filtered(current_df, user_keywords)

    def _generate_filename_options_filtered(self, current_df, user_keywords):
        all_tokens = []
        stop_words = {'电路图', '维修', '手册', '定义', '针脚', '保险', '原理图', '整车', '上汽', '红岩', '杰狮', '东风', '解放', '重汽'}
        
        for name in current_df['filename']:
            tokens = re.split(r'[^a-zA-Z0-9\u4e00-\u9fa5]+', str(name))
            for t in tokens:
                if len(t) > 1 and t not in stop_words and t not in user_keywords:
                    all_tokens.append(t)
        
        counter = Counter(all_tokens)
        valid_options = []
        for word, count in counter.most_common(10):
            if 1 < count < len(current_df):
                valid_options.append(word)
        
        if valid_options:
            return "请选择具体的型号或配置：", valid_options[:5]
        return None, []