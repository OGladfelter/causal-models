import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
d='/tmp/olist/'
o=pd.read_csv(d+'olist_orders_dataset.csv',parse_dates=['order_purchase_timestamp','order_delivered_customer_date','order_estimated_delivery_date'])
o=o[(o.order_status=='delivered')].dropna(subset=['order_delivered_customer_date'])
it=pd.read_csv(d+'olist_order_items_dataset.csv'); pr=pd.read_csv(d+'olist_products_dataset.csv')
tr=pd.read_csv(d+'product_category_name_translation.csv')
rv=pd.read_csv(d+'olist_order_reviews_dataset.csv'); cu=pd.read_csv(d+'olist_customers_dataset.csv'); se=pd.read_csv(d+'olist_sellers_dataset.csv')
ge=pd.read_csv(d+'olist_geolocation_dataset.csv').groupby('geolocation_zip_code_prefix')[['geolocation_lat','geolocation_lng']].mean()
it=it.merge(pr,on='product_id',how='left').merge(tr,on='product_category_name',how='left').merge(se,on='seller_id',how='left')
agg=it.groupby('order_id').agg(price=('price','sum'),freight=('freight_value','sum'),n_items=('order_item_id','max'),
    weight=('product_weight_g','sum'),category=('product_category_name_english','first'),
    seller_zip=('seller_zip_code_prefix','first'),seller_state=('seller_state','first'),n_sellers=('seller_id','nunique')).reset_index()
rvo=rv.groupby('order_id').review_score.mean().reset_index()   # mean if several
df=o.merge(agg,on='order_id').merge(rvo,on='order_id').merge(cu,on='customer_id')
df=df.join(ge.add_prefix('c_'),on='customer_zip_code_prefix').join(ge.add_prefix('s_'),on='seller_zip')
def hav(a,b,c,d_):
    a,b,c,d_=map(np.radians,[a,b,c,d_]); x=np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((d_-b)/2)**2
    return 6371*2*np.arcsin(np.sqrt(x))
df['distance_km']=hav(df.c_geolocation_lat,df.c_geolocation_lng,df.s_geolocation_lat,df.s_geolocation_lng)
df['late']=(df.order_delivered_customer_date>df.order_estimated_delivery_date).astype(int)
df['days_late']=(df.order_delivered_customer_date-df.order_estimated_delivery_date).dt.total_seconds()/86400
df['est_lead_days']=(df.order_estimated_delivery_date-df.order_purchase_timestamp).dt.total_seconds()/86400
df['month']=df.order_purchase_timestamp.dt.month; df['year']=df.order_purchase_timestamp.dt.year
df['same_state']=(df.customer_state==df.seller_state).astype(int)
df['high_review']=(df.review_score>=4).astype(int)
cols=['review_score','late','days_late','est_lead_days','price','freight','n_items','weight','n_sellers','distance_km','same_state','month','year','category','customer_state','seller_state']
df=df[cols].dropna(subset=['review_score','price','freight','weight','distance_km','category']).reset_index(drop=True)
df.to_csv('/tmp/work/olist_merged.csv',index=False)
print(df.shape, 'late share',df.late.mean().round(4))
print(df.groupby('late').review_score.agg(['mean','count']))
