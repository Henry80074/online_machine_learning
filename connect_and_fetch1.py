import psycopg2
import pandas as pd


def connect_and_fetch():

   #establishing the connection
   conn = psycopg2.connect(
      database="fear_and_greed", user='postgres', ***REMOVED***='224822', host='127.0.0.1', port= '5432'
   )
   sql_query = pd.read_sql_query('''
                                  SELECT
                                  *
                                  FROM fear_greed_index
                                  ''', conn)

   df = pd.DataFrame(sql_query, columns=['timestamp', 'value', 'price', 'value_classification'])

   #Closing the connection
   conn.close()
   return df