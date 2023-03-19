import pyterrier as pt

# terrier and pyterrier setup
pt.init()

# datasets

# ----- splade ------
import pyt_splade
query_pipeline = pyt_splade.SpladeFactory(agg='max').query()
df_expanded = query_pipeline(pd.DataFrame([
    {'qid': '1108939', 'query': 'what slows down the flow of blood'},
]))

# print(df_expanded['query'].values[0])