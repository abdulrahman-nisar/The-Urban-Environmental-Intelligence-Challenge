import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def run_data_cleaning_pipeline():
	df = pd.read_csv("./data/openaq_2025.csv")

	df = df.drop(columns=[
    	"country_iso", "isMobile", "isMonitor",
    	"location_name", "unit", "datetimeLocal",
    	"timezone", "owner_name", "provider"
	])

	df["parameter"] = df["parameter"].str.lower()

	variables = ["pm25", "pm10", "no2", "o3"]
	df = df[df["parameter"].isin(variables)]

	df_wide = df.pivot_table(
    	index=["location_id", "datetimeUtc", "longitude", "latitude"],
    	columns="parameter",
    	values="value"
	).reset_index()

	df_wide["pm25_raw"] = df_wide["pm25"]
	


	imputer = SimpleImputer(strategy="mean")
	df_wide[variables] = imputer.fit_transform(df_wide[variables])

	scaler = StandardScaler()
	df_wide[variables] = scaler.fit_transform(df_wide[variables])

	df_wide.to_csv("./data/output.csv")