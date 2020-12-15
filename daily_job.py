import requests
import io
import pandas as pd
import numpy as np
from datetime import date, timedelta

from source_code.backpip import data_up_pip
from source_code.SqlCo import Sqldd

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

tip = Sqldd()
cnx, cursor = tip.get_bdd_co()

this_date = date.today() - timedelta(days=1)
yesterday_date = date.today() - timedelta(days=2)
this_date = str(this_date)
yesterday_date = str(yesterday_date)

## scrap data from source
url="https://covid.ourworldindata.org/data/owid-covid-data.csv"
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))

## this dataset sometime no receive update, so i check the simple link 
verif_data = c[c['date'].isin([f"{this_date}"])]

if len(verif_data) == 0:
    print("Not today, check other link")
    url="https://covid.ourworldindata.org/data/ecdc/full_data.csv"
    s=requests.get(url).content
    c=pd.read_csv(io.StringIO(s.decode('utf-8')))
    
    verif_data = c[c['date'].isin([f"{this_date}"])]
    
    ## If with the second link data have not yet be updated, raise Error & try again later
    if len(verif_data) == 0:
        raise ValueError('Data have not receive Update at this moment, Try later !')
else:
    print("Scrap step Ok")
    

tini_df = c.query("date == @this_date")
tini_df = tini_df.replace({np.nan: 'Null'})

data_job = data_up_pip(tini_df) 
data_job.up_baseline()
data_job.clean_leave()

## Load data from days -1 to evaluate the performances of model #######################################################
data_load = pd.read_sql(f"SELECT * FROM cov_aipred WHERE date='{yesterday_date}';", con=cnx)

def Eval(c,data_load,country):

    """ function for evaluate all models
    params : 
        c = data scrap today
        data_load = dataset from day -1
        country = country concerned by eval
    """
    df = c
    df = df[df['location'].isin([f"{country}"])]
    df = df.reset_index()
    sle = df.iloc[-1]
    v0 = sle["date"]
    v1 = sle["total_cases"]
    v2 = sle["total_deaths"]
    
    res_tempo = data_load[data_load['country'].isin([f"{country}"])]
    res_tempo = res_tempo.reset_index()

    rez1 = res_tempo.at[0,"total_cases_predict"]
    rez2 = res_tempo.at[0,"total_deaths_predict"]
    
    errorCase1 = rez1-v1
    errorCase2 = rez2-v2

    follow_df = pd.DataFrame()
    follow_df.at[0,"date"] = v0
    follow_df.at[0,"country"] = country
    follow_df.at[0,"total_cases_predict"] = rez1
    follow_df.at[0,"total_cases_real"] = v1
    follow_df.at[0,"total_deaths_predict"] = rez2
    follow_df.at[0,"total_deaths_real"] = v2
    follow_df.at[0,"error_abs_cases"] = errorCase1
    follow_df.at[0,"error_abs_deaths"] = errorCase2
    
    return follow_df

## call eval function and store dataframe on var's
follow_df1 = Eval(c,data_load,"France")
follow_df2 = Eval(c,data_load,"China")
follow_df3 = Eval(c,data_load,"Italy")
# follow_df4 = Eval(c,data_load,"Belgium")
follow_df5 = Eval(c,data_load,"United States")
follow_df6 = Eval(c,data_load,"World")
follow_df7 = Eval(c,data_load,"United Kingdom")
follow_df8 = Eval(c,data_load,"Germany")
follow_df9 = Eval(c,data_load,"Iran")
follow_df10 = Eval(c,data_load,"Turkey")
follow_df11 = Eval(c,data_load,"Brazil")

## concatenate all df in 1
frames = [follow_df1, follow_df2, follow_df3,
          follow_df5, follow_df6, follow_df7, follow_df8,
          follow_df9, follow_df10, follow_df11]
rapport = pd.concat(frames)

## insert yesterday reporting to bdd
data_job = data_up_pip(rapport) 
data_job.up_rapp()
data_job.clean_leave()

df = pd.read_sql(f"SELECT * FROM cov_baseline;", con=cnx)

def createModel(df,subject,periode,country,n_splits=3,max_iter=5000):
    
    df = df[df['location'].isin([f"{country}"])] 
    df = df.reset_index()
    vals = df[[f"{subject}"]].values
    
    imp = SimpleImputer(missing_values=np.nan, strategy='constant')
    vals = imp.fit_transform(vals)
    
    hisShape = vals.shape[0]
    x_train = []
    y_train = []

    for i in range(periode,hisShape):
        x_train.append(vals[(i-periode):i,0]) 
        y_train.append(vals[i,0])

    x_train = np.array(x_train) 
    y_train = np.array(y_train)
    
    ###############################################################
    model = ElasticNet(random_state=0,max_iter=max_iter) 
    # tol=0.01 by reducing this hp warning disapear, pred will be highter 
    ###############################################################
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, x_train, y_train, cv=tscv)
    print(f"R^2: {scores.mean()} (+/- {scores.std()})")
    
    model.fit(x_train,y_train)
    print("Coef : ",model.coef_)
    
    return vals, x_train,y_train, model

def predicTomorow(vals,model,periode):
    tmw = vals[-periode:]
    lili = []
    for i in tmw:
        lili.append(int(i))
    lili = np.array(lili).reshape(1, -1) 
    res = int(model.predict(lili))
    last_day = lili[0][periode-1]
    diff_betw = res-last_day
    print(f"{periode} Last_days : ",lili)
    print("Today : ",last_day)
    print("Prediction + : ",diff_betw)
    print("Tomorow : ",res)
    return last_day, res

def conbine(df,subject,periode,country):
    vals, x_train, y_train, model = createModel(df,subject,periode,country)
    last_day, ress = predicTomorow(vals,model,periode)
    return last_day, ress

def fullRoutines(df,periode,country):
    print(f"********* For {country} **************")
    print("                                     ")
    print("--------- Total Cases -----------------")
    print("                                     ")
    last_day_cases, res1 = conbine(df,"total_cases",periode,f"{country}")
    print("                                     ")
    print("--------- Total Death -----------------")
    print("                                     ")
    last_day_death, res2 = conbine(df,"total_deaths",periode,f"{country}")
    print("                                     ")
    return last_day_cases, last_day_death, res1, res2

periode = 3

last_day_cases, last_day_death, res1ww, res2ww = fullRoutines(df,periode,"World")
world_data = last_day_cases, last_day_death, res1ww, res2ww

last_day_casesfr, last_day_deathfr, res1fr, res2fr = fullRoutines(df,periode,"France")
french_data = last_day_casesfr, last_day_deathfr, res1fr, res2fr

last_day_casesch, last_day_deathch, res1ch, res2ch = fullRoutines(df,periode,"China")
china_data = last_day_casesch, last_day_deathch, res1ch, res2ch

last_day_casesit, last_day_deathit, res1it, res2it = fullRoutines(df,periode,"Italy")
italy_data = last_day_casesit, last_day_deathit, res1it, res2it


# last_day_casessp, last_day_deathsp, res1sp, res2sp = fullRoutines(df,periode,"Belgium")
# spain_data = last_day_casessp, last_day_deathsp, res1sp, res2sp


last_day_casesus, last_day_deathus, res1us, res2us = fullRoutines(df,periode,"United States")
usa_data = last_day_casesus, last_day_deathus, res1us, res2us

last_day_casesuk, last_day_deathuk, res1uk, res2uk = fullRoutines(df,periode,"United Kingdom")
uk_data = last_day_casesuk, last_day_deathuk, res1uk, res2uk

last_day_casesger, last_day_deathger, res1ger, res2ger = fullRoutines(df,periode,"Germany")
ger_data = last_day_casesger, last_day_deathger, res1ger, res2ger

last_day_casesIran, last_day_deathIran, res1Iran, res2Iran = fullRoutines(df,periode,"Iran")
Iran_data = last_day_casesIran, last_day_deathIran, res1Iran, res2Iran

last_day_casesTurk, last_day_deathTurk, res1Turk, res2Turk = fullRoutines(df,periode,"Turkey")
Turk_data = last_day_casesTurk, last_day_deathTurk, res1Turk, res2Turk

last_day_casesBraz, last_day_deathBraz, res1Braz, res2Braz = fullRoutines(df,periode,"Brazil")
Braz_data = last_day_casesBraz, last_day_deathBraz, res1Braz, res2Braz

def popPred(country,rez1,rez2):
    pop_pred = pd.DataFrame()
    pop_pred.loc[0,"date"] = this_date
    pop_pred.loc[0,"country"] = country
    pop_pred.loc[0,"total_cases_predict"] = rez1
    pop_pred.loc[0,"total_cases_real"] = 0
    pop_pred.loc[0,"total_deaths_predict"] = rez2
    pop_pred.loc[0,"total_deaths_real"] = 0
    pop_pred.loc[0,"error_abs_cases"] = 0
    pop_pred.loc[0,"error_abs_deaths"] = 0
    return pop_pred

df_pop_pred1 = popPred("France",res1fr, res2fr)
df_pop_pred2 = popPred("China",res1ch, res2ch)
df_pop_pred3 = popPred("Italy",res1it, res2it)
# df_pop_pred4 = popPred("Belgium",res1sp, res2sp)
df_pop_pred5 = popPred("United States",res1us, res2us)
df_pop_pred6 = popPred("World",res1ww, res2ww)
df_pop_pred7 = popPred("United Kingdom",res1uk, res2uk)
df_pop_pred8 = popPred("Germany",res1ger, res2ger)
df_pop_pred9 = popPred("Iran",res1Iran, res2Iran)
df_pop_pred10 = popPred("Turkey",res1Turk, res2Turk)
df_pop_pred11 = popPred("Brazil",res1Braz, res2Braz)

frames = [df_pop_pred1, df_pop_pred2, df_pop_pred3,
          df_pop_pred5, df_pop_pred6, df_pop_pred7, df_pop_pred8,
          df_pop_pred9, df_pop_pred10, df_pop_pred11]

predpred = pd.concat(frames)

pre_pred_tobdd = predpred.drop(columns=['total_cases_real','total_deaths_real','error_abs_cases','error_abs_deaths'])
data_job = data_up_pip(pre_pred_tobdd) 
data_job.up_pred()
data_job.clean_leave()