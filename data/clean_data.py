import pandas as pd
from datetime import date
import numpy as np

covid = pd.read_csv("covid.csv")
census = pd.read_csv("census.csv")


#Clean the census data
census = census[census['State'] != "Puerto Rico"]
census = census[census['State'] != "District of Columbia"]
census["GenderRatio"] = census["Men"]/(census["Men"] + census["Women"])
census = pd.concat([census, pd.get_dummies(census['State'], prefix = 'State', drop_first = True)], axis=1).drop(['State'], axis = 1)
drop_labels = ["IncomeErr", "IncomePerCapErr", "Production", "OtherTransp", "FamilyWork", "Men", "Women"]
census.drop(drop_labels, 'columns', inplace = True)
headers_to_mod = ["Hispanic", "White", "Black", "Native", "Asian", "Pacific", "VotingAgeCitizen", "Employed"]
for h in headers_to_mod:
    census[h] /= census["TotalPop"]

covid = covid[~pd.isnull(covid.fips)]

m_covid = len(covid["date"])
infected_counties = covid[covid.date == covid.date[m_covid-1]]
infected_counties = infected_counties[infected_counties["cases"] >= 50]

combined = census[census.CountyId.isin(infected_counties.fips)]
infected_counties = infected_counties[infected_counties.fips.isin(combined.CountyId)]


days = []
for i in range(len(infected_counties)):
    this_county = covid[covid["fips"] == infected_counties.fips.iloc[i]]
    five_date = this_county.date[this_county[this_county.cases >= 5].first_valid_index()]
    fifty_date = this_county.date[this_county[this_county.cases >= 50].first_valid_index()]
    diff = (date.fromisoformat(fifty_date) - date.fromisoformat(five_date)).days
    days.append(diff)
combined["days"] = days
combined.to_csv("combined.csv")

