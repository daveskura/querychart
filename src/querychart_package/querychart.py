"""
  Dave Skura, 2023
"""
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from schemawizard_package.schemawizard import schemawiz

def main():
	print('matplotlib.__version__ = ',matplotlib.__version__)
	#obj = charter()
	query = """
	SELECT '2023/03/29' as caldt, 10.99 as sales_amt, 8.55 as cost
	UNION
	SELECT '2023/03/30' as caldt, 15.99 as sales_amt, 18.45 as cost
	UNION
	SELECT '2023/03/31' as caldt, 13.99 as sales_amt, 15.35 as cost
	UNION
	SELECT '2023/04/01' as caldt, 17.99 as sales_amt, 19.55 as cost
	"""
	#obj.querychart('graph query columns eg1',obj.schwiz.dbthings.sqlite_db,query)

	sql = """
	SELECT 'sales_amt' as legend, 10.99 as "2023/03/29", 15.99 as "2023/03/30" , 13.99 as "2023/03/31", 17.99 as "2023/04/01" 
	UNION
	SELECT 'cost' as legend, 8.55 as "2023/03/29", 18.45  as "2023/03/30" ,  15.35 as "2023/03/31", 19.55 as "2023/04/01" 
	"""
	#obj.widequerychart('graph query columns eg2',obj.schwiz.dbthings.sqlite_db,sql)

	#charter().csv_querychart('sales.csv','SELECT cal_dt,sales_amt,cost FROM sales.csv ORDER BY cal_dt')
	#charter().csv_widequerychart('widesales.csv',"SELECT legend," + '"2023/03/29","2023/03/30","2023/03/31","2023/04/01" FROM widesales.csv')


	#charter().showcsv('sales.csv','SELECT cal_dt FROM sales.csv ORDER BY cal_dt')
	#charter().showcsv('widesales.csv','SELECT * FROM widesales.csv')

class charter():
	def __init__(self):
		self.schwiz = schemawiz()


	#
	# SELECT legend,"2023/03/29","2023/03/30","2023/03/31","2023/04/01" FROM sales.csv 
	#
	# legend    2023/03/29 2023/03/30 2023/03/31 2023/04/01 
	# --------- ---------- ---------- ---------- ---------- 
	# cost      8.55       18.45      15.35      19.55      
	# sales_amt 10.99      15.99      13.99      17.99      
	#
	def csv_widequerychart(self,csv_filename,query,xlabel='points',ylabel='Amounts'):
		self.schwiz.createload_sqlite_from_csv(csv_filename)
		tablename = self.schwiz.lastcall_tablename
		newquery = query.replace(csv_filename,tablename)
		title = csv_filename

		self.widequerychart(title,self.schwiz.dbthings.sqlite_db,newquery,xlabel,ylabel)

		self.schwiz.dbthings.sqlite_db.execute('DROP TABLE ' + tablename)

	#
	# SELECT cal_dt,sales_amt,cost FROM sales.csv ORDER BY cal_dt 
	#
	# cal_dt,product,customer,sales_amt,cost
	# 2023/03/29,food,dave,10.99,8.55
	# 2023/03/30,electronics,dave,15.99,18.45
	# 2023/03/31,booze,dave,20.99,2.35
	# 2023/04/01,food,dave,5.99,43.25
	#
	def csv_querychart(self,csv_filename,query,xlabel='',ylabel='Amounts'):
		self.schwiz.createload_sqlite_from_csv(csv_filename)
		tablename = self.schwiz.lastcall_tablename
		newquery = query.replace(csv_filename,tablename)
		title = csv_filename

		self.querychart(title,self.schwiz.dbthings.sqlite_db,newquery)

		self.schwiz.dbthings.sqlite_db.execute('DROP TABLE ' + tablename)


	# widequerychart charts each row as a line
	#
	# SELECT 'sales_amt' as legend, 10.99 as "2023/03/29", 15.99 as "2023/03/30" , 13.99 as "2023/03/31", 17.99 as "2023/04/01" 
	# UNION
	# SELECT 'cost' as legend, 8.55 as "2023/03/29", 18.45  as "2023/03/30" ,  15.35 as "2023/03/31", 19.55 as "2023/04/01" 
	#
	# legend    2023/03/29 2023/03/30 2023/03/31 2023/04/01 
	# --------- ---------- ---------- ---------- ---------- 
	# cost      8.55       18.45      15.35      19.55      
	# sales_amt 10.99      15.99      13.99      17.99      
	def widequerychart(self,title,db,query,xlabel='points',ylabel='Amounts'):
		plt.title(title)
		plt.ylabel(ylabel)
		data = []
		xaxis = []

		print(query,'\n')
		print(db.export_query_to_str(query))

		onerundata = db.query(query)
		rowcount = 0
		column_names = []
		legend_names = []
		for k in [i[0] for i in onerundata.description]:
			column_names.append(k)

		for i in range(1,len(column_names)):
			xaxis.append(column_names[i])

		plt.xlabel(xlabel)

		for row in onerundata:
			legend_names.append(row[0])

			data.append(row)
			rowcount += 1
			colcount = len(row)

		yaxis = [['' for x in range(colcount-1)] for y in range(rowcount)] 
		irow = 0
		for row in data:
			for i in range(0,colcount):
				if i > 0:
					yaxis[irow][i-1] = row[i]
			irow += 1

		for i in range(0,rowcount):
			xpoints = np.array(xaxis)
			ypoints = np.array(yaxis[i])
			plt.plot(xpoints, ypoints, marker = 'o')

		
		tickdisplaycount = int(colcount/10) + 1
		plt.xticks(xpoints[::tickdisplaycount],  rotation=45)

		# Add margins (padding) so that markers don't get clipped by the axes
		plt.margins(0.05)

		plt.legend(legend_names, loc=0, frameon=True)

		plt.show()

	# querychart charts each column as a line, with column 1 as the xaxis
	#
	# eg 1.
	# SELECT caldt,sales_amt,cost
	# FROM sales.csv
	#
	# eg 2
	# SELECT '2023/03/29' as caldt, 10.99 as sales_amt, 8.55 as cost
	# UNION
	# SELECT '2023/03/30' as caldt, 15.99 as sales_amt, 18.45 as cost
	# UNION
	# SELECT '2023/03/31' as caldt, 13.99 as sales_amt, 15.35 as cost
	# UNION
	# SELECT '2023/04/01' as caldt, 17.99 as sales_amt, 19.55 as cost
	#
	# caldt      sales_amt cost  
	# ---------- --------- ----- 
	# 2023/03/29 10.99     8.55  
	# 2023/03/30 15.99     18.45 
	# 2023/03/31 13.99     15.35 
	# 2023/04/01 17.99     19.55 
	# 

	def querychart(self,title,db,query,xlabel='',ylabel='Amounts'):
		plt.title(title)
		plt.ylabel(ylabel)
		data = []
		xaxis = []

		print(query,'\n')
		print(db.export_query_to_str(query))

		onerundata = db.query(query)
		rowcount = 0
		column_names = []
		legend_names = []
		skippedone = False
		for k in [i[0] for i in onerundata.description]:
			column_names.append(k)
			if skippedone:
				legend_names.append(k)
			else:
				skippedone = True

		if xlabel != '':
			plt.xlabel(xlabel)
		else:
			plt.xlabel(column_names[0])

		for row in onerundata:
			data.append(row)
			rowcount += 1
			colcount = len(row)

		# Creates a list containing 5 lists, each of 8 items, all set to 0
		yaxis = [['' for x in range(rowcount)] for y in range(colcount-1)] 
		irow = 0
		for row in data:
			for i in range(0,colcount):
				if i == 0:
					xaxis.append(row[i])
				else:
					yaxis[i-1][irow] = row[i]
			irow += 1

		for i in range(0,colcount-1):
			xpoints = np.array(xaxis)
			ypoints = np.array(yaxis[i])
			plt.plot(xpoints, ypoints, marker = 'o')

		
		tickdisplaycount = int(rowcount/10) + 1
		plt.xticks(xpoints[::tickdisplaycount],  rotation=45)

		# Add margins (padding) so that markers don't get clipped by the axes
		plt.margins(0.05)

		plt.legend(legend_names, loc=0, frameon=True)


		plt.show()


	def showcsv(self,csv_filename,showquery=''):
		self.schwiz.createload_sqlite_from_csv(csv_filename)
		tablename = self.schwiz.lastcall_tablename
		if showquery == '':
			query = 'SELECT * FROM ' + tablename
		else:
			query = showquery.replace(csv_filename,tablename)

		print(self.schwiz.dbthings.sqlite_db.export_query_to_str(query))

		self.schwiz.dbthings.sqlite_db.execute('DROP TABLE ' + tablename)

	def demo1(self):
		xpoints = np.array([0, 6])
		ypoints = np.array([0, 250])

		plt.plot(xpoints, ypoints)
		plt.show()

	def demo2(self):
		ypoints = np.array([3, 8, 1, 10])

		# marker ref 
		# https://www.w3schools.com/python/matplotlib_markers.asp
		plt.plot(ypoints, marker = 'X')
		plt.show()

	def demo3(self):
		ypoints = np.array([3, 8, 1, 10])

		#plt.plot(ypoints, 'o:r', ms = 5)
		#plt.plot(ypoints, marker = 'o', ms = 20, mfc = 'r')
		#plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r', mfc = 'r')
		#plt.plot(ypoints, marker = 'o', ms = 20, mec = '#4CAF50', mfc = '#4CAF50')
		#plt.plot(ypoints, marker = 'o', ms = 20, mec = 'hotpink', mfc = 'hotpink')
		#plt.plot(ypoints, linestyle = 'dashed')
		#plt.plot(ypoints, ls = '--')
		#plt.plot(ypoints, linestyle = 'dotted')
		#plt.plot(ypoints, ls = ':')
		plt.plot(ypoints, color = 'r')
		plt.show()

	def demo4(self):
		y1 = np.array([3, 8, 1, 10])
		y2 = np.array([6, 2, 7, 11])

		plt.plot(y1)
		plt.plot(y2)

		plt.show()

	def demo5(self):

		x1 = np.array([0, 1, 2, 3])
		y1 = np.array([3, 8, 1, 10])
		x2 = np.array([0, 1, 2, 3])
		y2 = np.array([6, 2, 7, 11])

		plt.plot(x1, y1, x2, y2)
		plt.show()

	def demo6(self):

		x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
		y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

		font1 = {'family':'serif','color':'blue','size':20}
		font2 = {'family':'serif','color':'darkred','size':15}

		plt.title("Sports Watch Data", fontdict = font1,loc = 'center')
		plt.xlabel("Average Pulse", fontdict = font2)
		plt.ylabel("Calorie Burnage", fontdict = font2)

		plt.plot(x, y)
		plt.grid()
		#plt.grid(axis = 'y')
		#plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
		plt.show()

	
	def demo7(self):

		#plot 1:
		x = np.array([0, 1, 2, 3])
		y = np.array([3, 8, 1, 10])

		plt.subplot(1, 2, 1)
		plt.plot(x,y)
		plt.title("SALES")

		#plot 2:
		x = np.array([0, 1, 2, 3])
		y = np.array([10, 20, 30, 40])

		plt.subplot(1, 2, 2)
		plt.plot(x,y)
		plt.title("INCOME")

		plt.suptitle("MY SHOP")
		plt.show()

	def demo8(self):
		x = np.array(["A", "B", "C", "D"])
		y = np.array([3, 8, 1, 10])

		#plt.barh(x, y)
		#plt.barh(x, y, height = 0.1)

		#plt.bar(x,y)
		#plt.bar(x, y, color = "red")
		plt.bar(x, y, width = 0.1)

		plt.show()

	def demo9(self):
		y = np.array([35, 25, 25, 15])
		mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
		myexplode = [0.2, 0, 0, 0]

		plt.pie(y, labels = mylabels, explode = myexplode, shadow = True)
		plt.legend(title = "Four Fruits:")
		plt.show() 


if __name__ == '__main__':
	main()

