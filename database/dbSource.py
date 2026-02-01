import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

db_url = f'mysql+pymysql://{os.getenv("dbuser")}:{os.getenv("dbpassword")}@{os.getenv("dbhost")}:{os.getenv("dbport")}/{os.getenv("dbname")}'

engine = create_engine(db_url)

session = sessionmaker(bind=engine)

session()



queries = {

    "admissions": """ SELECT * FROM admissions ; """,
    "daily_metrics": """ SELECT * FROM daily_metrics; """,
    "hospitals": """ SELECT * FROM hospitals; """,
    "wards": """ SELECT * FROM wards; """

}



try:

    for table, query in queries.items():

        result = pd.read_sql(query,engine)

        file_path = os.path.join("dataset", f"{table}.csv")

        result.to_csv(file_path, index=False)

except Exception as e:
    print(f"An error occurred: {str(e)}")
   


