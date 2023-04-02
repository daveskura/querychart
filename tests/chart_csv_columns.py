"""
  Dave Skura
"""
from querychart_package.querychart import charter

obj = charter()

#charter().showcsv('sales.csv','SELECT * FROM sales.csv ORDER BY cal_dt')
charter().csv_querychart('sales.csv','SELECT cal_dt,sales_amt,cost FROM sales.csv ORDER BY cal_dt')
