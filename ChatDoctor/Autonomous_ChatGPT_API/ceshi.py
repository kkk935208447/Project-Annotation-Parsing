import pandas as pd
df = pd.read_csv("./disease_database_mini.csv")
divided_text = []
# 将CSV数据分成多个文本块
csvdata = df.to_dict('records')
print(csvdata)
step_length = 15
for csv_item in range(0,len(csvdata),step_length):
    csv_text = str(csvdata[csv_item:csv_item+step_length]).replace("}, {", "\n\n").replace("\"", "")#.replace("[", "").replace("]", "")
    divided_text.append(csv_text)
print(divided_text)