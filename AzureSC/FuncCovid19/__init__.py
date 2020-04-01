import datetime
from datetime import date, timedelta
import logging

import requests
import io

import azure.functions as func

import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import TimeSeriesSplit

def Eval(c,data_load,country):
    df = c
    df = df[df['location'].isin([f"{country}"])]
    sle = df.iloc[-1]
    v0 = sle["date"]
    v1 = sle["total_cases"]
    v2 = sle["total_deaths"]
    
    res_tempo = data_load[data_load['country'].isin([f"{country}"])]
    rez1 = res_tempo.loc[0,"total_cases_predict"]
    rez2 = res_tempo.loc[0,"total_deaths_predict"]
    
    errorCase1 = rez1-v1
    errorCase2 = rez2-v2

    follow_df = pd.DataFrame()
    follow_df.loc[0,"date"] = v0
    follow_df.loc[0,"country"] = country
    follow_df.loc[0,"total_cases_predict"] = rez1
    follow_df.loc[0,"total_cases_real"] = v1
    follow_df.loc[0,"total_deaths_predict"] = rez2
    follow_df.loc[0,"total_deaths_real"] = v2
    follow_df.loc[0,"error_abs_cases"] = errorCase1
    follow_df.loc[0,"error_abs_deaths"] = errorCase2
    
    return follow_df

def createModel(subject,periode,country,n_splits=3,max_iter=5000):
    
    df = pd.read_csv(f".\\Base_Files\\full_data{date.today()}.csv")
    df = df[df['location'].isin([f"{country}"])] 
    vals = df[[f"{subject}"]].values
    
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
    return res

def conbine(subject,periode,country):
    vals, x_train, y_train, model = createModel(subject,periode,country)
    ress = predicTomorow(vals,model,periode)
    return ress

def fullRoutines(df,periode,country):
    print(f"********* For {country} **************")
    print("                                     ")
    print("--------- Total Cases -----------------")
    print("                                     ")
    res1 = conbine("total_cases",periode,f"{country}")
    print("                                     ")
    print("--------- Total Death -----------------")
    print("                                     ")
    res2 = conbine("total_deaths",periode,f"{country}")
    print("                                     ")
    return res1, res2

def popPred(country,rez1,rez2):
    pop_pred = pd.DataFrame()

    pop_pred.loc[0,"date"] = date.today()
    pop_pred.loc[0,"country"] = country
    pop_pred.loc[0,"total_cases_predict"] = rez1
    pop_pred.loc[0,"total_cases_real"] = 0
    pop_pred.loc[0,"total_deaths_predict"] = rez2
    pop_pred.loc[0,"total_deaths_real"] = 0
    pop_pred.loc[0,"error_abs_cases"] = 0
    pop_pred.loc[0,"error_abs_deaths"] = 0
    
    return pop_pred

def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()

    if mytimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function ran at %s', utc_timestamp)

    url="https://covid.ourworldindata.org/data/ecdc/full_data.csv"
    s=requests.get(url).content
    c=pd.read_csv(io.StringIO(s.decode('utf-8')))
    c.to_csv(f".\\Base_Files\\full_data{date.today()}.csv")
    data_load = pd.read_csv(f".\\Pred\\predDf{date.today() - timedelta(days=1)}.csv", index_col=0)

    follow_df1 = Eval(c,data_load,"France")
    follow_df2 = Eval(c,data_load,"China")
    follow_df3 = Eval(c,data_load,"Italy")
    follow_df4 = Eval(c,data_load,"Spain")
    follow_df5 = Eval(c,data_load,"United States")

    frames = [follow_df1,follow_df2,follow_df3,follow_df4,follow_df5]
    rapport = pd.concat(frames)
    rapport.to_csv(f".\\Rapport\\rap{date.today()}.csv")

    df = pd.read_csv(f".\\Base_Files\\full_data{date.today()}.csv")

    periode = 3

    res1fr, res2fr = fullRoutines(df,periode,"France")
    res1ch, res2ch = fullRoutines(df,periode,"China")
    res1it, res2it = fullRoutines(df,periode,"Italy")
    res1sp, res2sp = fullRoutines(df,periode,"Spain")
    res1us, res2us = fullRoutines(df,periode,"United States")

    df_pop_pred1 = popPred("France",res1fr, res2fr)
    df_pop_pred2 = popPred("China",res1ch, res2ch)

    df_pop_pred3 = popPred("Italy",res1it, res2it)
    df_pop_pred4 = popPred("Spain",res1sp, res2sp)
    df_pop_pred5 = popPred("United States",res1us, res2us)

    frames = [df_pop_pred1,df_pop_pred2,df_pop_pred3,df_pop_pred4,df_pop_pred5]
    predpred = pd.concat(frames)

    predpred.to_csv(f".\\Pred\\predDf{date.today()}.csv")

















