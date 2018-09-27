import pandas as pd;
import sqlite3 as sq

dat = sq.connect('./data/MotionGesture.db')


#query = dat.execute("SELECT * From GestureTable")

#cols = [column[0] for column in query.description]

#results = pd.DataFrame.from_records(data = query.fetchall(), columns= cols)


df = pd.read_sql_query("SELECT * From GestureTable", dat)

df.to_csv('./Data/converted_database.csv')

print(df)

#raw_data = {'x_pos,': [],
 #           'y_pos':[],
  #          'z_pos':[],
   #         'pitch':[],
    #        'yaw':[],
     #       'roll':[]

      #      }
#df = pd.DataFrame(raw_data, columns=['X Position', 'Y Position', 'Z Position', 'Pitch', 'Yaw', ])