## Covid-19 : IA Machine Learning experiment

## Full application for windows user :

    Just click on the Covid Predictor.exe !
    In the soft, set data's for 3 last days of your location or for the world and click on prediction,
    Ia model will display a prediction for tomorow.

    Application is in pre-alpha state and work only on windows setup.

## Ia experiment :

(You will find prediction for tomorow & reporting every day at notebook 03 : 
 https://github.com/Nico-Facto/Covid-19/blob/master/03-Visu-Over-Time.ipynb)

With this project i try to predict number of cases & death the day before, for the next day.

This project follow 5 country :
    
    -France
    -China
    -Italy
    -Spain
    -Usa

but we can do this with all country.

Sources of data : https://ourworldindata.org/coronavirus-source-data
i take the full data

## Run experiment :

    First launch the notebook 0 , you will have the score of prediction made the day before
    and create new dataset to predcict tomorow.
    then run notebook 01.

## Next step :

    -Dockerise the project
    -Working on the model


## Some Note's

    -I'am a student in Ml, so i work on the Pipeline, but actualy 29/03, i don't really work on model's,
     I'm looking for the moment has automated the project, when I am satisfied with this work 
     I will work on the model

## Folders & data :

    All those files are generated by scripts, you dont have to open it. They are on AzureSC folder

    Base_Files = full data donwload by the script 01
    pred = Predition made at date (file names contains date)
    rapport = evaluation predcit vs real with error abs.

## Prediction For tomorow 02/04 :
    
    date of prediction will be day -1.

        date	country	        total_cases_predict		total_deaths_predict
    2020-04-01	France	             59308.0		        4099.0	
    2020-04-01	China	             82497.0		        3311.0
    2020-04-01	Italy	             109477.0		        13273.0
    2020-04-01	Spain	             103481.0		        8907.0
    2020-04-01	United States	     216496.0		        5062.0

## Reporting : 

    Follow daily reporting on the notebook 03 : https://github.com/Nico-Facto/Covid-19/blob/master/03-Visu-Over-Time.ipynb




     






