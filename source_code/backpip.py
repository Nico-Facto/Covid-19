from source_code.SqlCo import Sqldd
import pandas as pd

class data_up_pip():

    def __init__(self, inc_df):
        tip = Sqldd()
        self.cnx, self.cursor = tip.get_bdd_co()
        self.main_df = inc_df

    def up_baseline(self):
        df = self.main_df
        for i,row in df.iterrows():
            value_tuple = (row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],
                        row[11],row[12],row[13],row[14],row[15],row[16],row[17],row[18],row[19],row[20],row[21],
                        row[22],row[23],row[24],row[25],row[26],row[27],row[28],row[29],row[30],row[31],row[32],
                        row[33],row[34],row[35],row[36],row[37],row[38],row[39],row[40])
            self.cursor.execute(f"""INSERT INTO cov_baseline (iso_code, continent, location, date, total_cases, new_cases,
                                                                new_cases_smoothed, total_deaths, new_deaths,
                                                                new_deaths_smoothed, total_cases_per_million,
                                                                new_cases_per_million, new_cases_smoothed_per_million,
                                                                total_deaths_per_million, new_deaths_per_million,
                                                                new_deaths_smoothed_per_million, new_tests, total_tests,
                                                                total_tests_per_thousand, new_tests_per_thousand,
                                                                new_tests_smoothed, new_tests_smoothed_per_thousand,
                                                                tests_per_case, positive_rate, tests_units, stringency_index,
                                                                population, population_density, median_age, aged_65_older,
                                                                aged_70_older, gdp_per_capita, extreme_poverty,
                                                                cardiovasc_death_rate, diabetes_prevalence, female_smokers,
                                                                male_smokers, handwashing_facilities, hospital_beds_per_thousand,
                                                                life_expectancy, human_development_index)
                             VALUES {value_tuple};""")

            self.cnx.commit()
        print(" Insert Baseline Done !! ")

    def up_pred(self):
        df = self.main_df
        for i,row in df.iterrows():
            cv_date = str(row[0])
            value_tuple = (cv_date,row[1],row[2],row[3])
            self.cursor.execute(f"""INSERT INTO cov_aipred (date, country, total_cases_predict, total_deaths_predict)
                             VALUES {value_tuple};""")
            self.cnx.commit()
        print(" Insert Prediction Done !! ")

    def up_rapp(self):
        df = self.main_df
        for i,row in df.iterrows():
            value_tuple = (row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7])
            self.cursor.execute(f"""INSERT INTO cov_rapport (date, country, total_cases_predict, total_cases_real,
            total_deaths_predict, total_deaths_real, error_abs_cases, error_abs_deaths)
                             VALUES {value_tuple};""")
            self.cnx.commit()
        print(" Insert Rapport Done !! ")

    def clean_leave(self):
        self.cnx.commit()
        self.cnx.close()
        self.cursor.close()


