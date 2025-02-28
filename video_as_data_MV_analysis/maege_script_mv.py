#最終的に異なるシーンdetectでベクトルを作った場合h以下を参考にidを軸に結合処理をして。
df_pv= pd.merge(
    df_mvs,
    df_merged[['id', 'pv_vector']],
    on='id',
    how='left'
)
df_pv