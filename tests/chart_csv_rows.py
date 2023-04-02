"""
  Dave Skura
"""
from querychart_package.querychart import charter

obj = charter()

#charter().showcsv('widesales.csv','SELECT * FROM widesales.csv')
charter().csv_widequerychart('widesales.csv','SELECT * FROM widesales.csv')
