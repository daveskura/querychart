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
	#charter().csv_querychart('sales.csv','SELECT cal_dt,sales_amt,cost FROM sales.csv ORDER BY cal_dt')
	#charter().showcsv('sales.csv','SELECT cal_dt FROM sales.csv ORDER BY cal_dt')
	#charter().showcsv('sales.csv')

class charter():
	def __init__(self):
		self.schwiz = schemawiz()

	def csv_querychart(self,csv_filename,query):
		self.schwiz.createload_sqlite_from_csv(csv_filename)
		tablename = self.schwiz.lastcall_tablename
		newquery = query.replace(csv_filename,tablename)
		title = csv_filename

		self.querychart(title,self.schwiz.dbthings.sqlite_db,newquery)

		self.schwiz.dbthings.sqlite_db.execute('DROP TABLE ' + tablename)

	# first column is x axis points. suggest dates.
	def querychart(self,title,db,query):
		plt.title(title)
		plt.ylabel('Amounts')
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

