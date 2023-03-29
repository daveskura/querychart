"""
  Dave Skura
  
  File Description:
"""
from querychart_package.querychart import charter

print (" Starting ") # 
obj = charter()

#obj.showcsv('sales.csv','SELECT cal_dt,sales_amt,cost FROM sales.csv ORDER BY cal_dt')
obj.csv_querychart('sales.csv','SELECT cal_dt,sales_amt,cost FROM sales.csv ORDER BY cal_dt')
