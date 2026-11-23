import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from econml.dml import LinearDML
df=pd.read_csv('/tmp/work/olist_merged.csv')
top=df.category.value_counts().index[:15]; df['cat']=np.where(df.category.isin(top),df.category,'other')
num=['est_lead_days','price','freight','n_items','weight','n_sellers','distance_km','same_state']
X=pd.concat([df[num],pd.get_dummies(df[['cat','customer_state','seller_state']],drop_first=True).astype(float),
             pd.get_dummies(df[['month','year']].astype(str),drop_first=True).astype(float)],axis=1)
Y=df.review_score.values; T=df.late.values
print('naive', Y[T==1].mean()-Y[T==0].mean())
# OLS adjusted
f='review_score ~ late + est_lead_days + np.log1p(price)+np.log1p(freight)+n_items+np.log1p(weight)+n_sellers+np.log1p(distance_km)+same_state+C(cat)+C(customer_state)+C(seller_state)+C(month)+C(year)'
m=smf.ols(f,df).fit(cov_type='HC1'); print('OLS',m.params['late'],m.conf_int().loc['late'].tolist())
# propensity / overlap / IPW
from sklearn.model_selection import cross_val_predict
ps=cross_val_predict(HistGradientBoostingClassifier(max_depth=4,max_iter=150,random_state=0),X,T,cv=5,method='predict_proba')[:,1]
print('ps range',ps.min().round(3),ps.max().round(3),'ps pct in [0.01,0.99]',((ps>0.01)&(ps<0.99)).mean().round(4))
psc=np.clip(ps,0.01,0.99); w=T/psc+(1-T)/(1-psc)
ipw=np.average(Y[T==1],weights=w[T==1])-np.average(Y[T==0],weights=w[T==0]); print('IPW',ipw)
# DML
def dml(Tv,seed=0):
    e=LinearDML(model_y=HistGradientBoostingRegressor(max_depth=4,max_iter=150,random_state=seed),
                model_t=HistGradientBoostingClassifier(max_depth=4,max_iter=150,random_state=seed),
                discrete_treatment=True,cv=5,random_state=seed)
    e.fit(Y,Tv,X=None,W=X.values); return e
e=dml(T); ate=float(e.ate_[0]) if hasattr(e,'ate_') else None
print('DML ate',e.const_marginal_ate(), e.const_marginal_ate_interval(alpha=0.05))
# placebo
rng=np.random.default_rng(0); pl=[]
for s in range(3):
    ep=dml(rng.permutation(T),seed=s); pl.append(float(np.ravel(ep.const_marginal_ate())[0]))
print('placebo',pl)
# CATE by distance tercile via LinearDML with X=distance
Xd=df[['distance_km']].values
e2=LinearDML(model_y=HistGradientBoostingRegressor(max_depth=4,max_iter=150,random_state=0),
             model_t=HistGradientBoostingClassifier(max_depth=4,max_iter=150,random_state=0),discrete_treatment=True,cv=5,random_state=0)
e2.fit(Y,T,X=Xd,W=X.drop(columns=['distance_km']).values)
q=df.distance_km.quantile([.1,.5,.9]).values; print('dist q',q)
print('CATE at 10/50/90 pct distance',e2.effect(q.reshape(-1,1)))
# subgroup naive-adjusted? same_state
for g in [0,1]:
    s=df.same_state==g; mm=smf.ols(f.replace('+same_state',''),df[s]).fit(); print('OLS same_state',g,mm.params['late'],s.sum(), df[s].late.mean())
