"""
  Dave Skura
	https://towardsdatascience.com/gantt-charts-with-pythons-matplotlib-395b7af72d72

  
"""
import sys
import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pandas import Timestamp
from schemawizard_package.schemawizard import schemawiz
def do_main():
	sql = """
		SELECT 'a Title' as Title	/*-- field[0] = Title */
				, Task								  -- field[1] = Task Name
				, Stream							  -- field[2] = Task Group
				, Start_dt						  -- field[3] = Start Date
				, End_dt							  -- field[4] = End Date
				, Completionpct				  -- field[5] = Completion percent from 0.0 - 1.0
		FROM sampledata.csv
	"""
	gantter().gantt_From_sqlcsv('sampledata.csv',sql)

	#gantter().show_how_to() #
	#gantter().gantt_From_csv('sampledata.csv','a title','Task','Stream','Start_dt','End_dt','Completionpct','%m/%d/%Y')
	#gantter().gantt_demo()

class gantter():
	def __init__(self,gantdatacsvfilename = ''): # data.csv
		print(" chumbo ") #
		self.schwiz = schemawiz()
		self.tasknamefield = 'Task'
		self.taskgroup = 'Department'
		self.startfield = 'Start'
		self.endfield = 'End'
		self.completionfield = 'Completion'
		self.datafile = gantdatacsvfilename	
		self.colorlist = ['#E64646','#E69646', '#34D05C', '#34D0C3', '#3475D0','#336600','#663300','#990000','#3300CC','#CC0000']
		self.bars = []
		self.barcolors = {}
		self.df = ''
		
	def show_how_to(self):
		notes = """
data expected to look as follows:

	Task,Stream,Start_dt,End_dt,Completionpct,Phases
	TSK M,IT,3/17/2022,3/20/2022,0.0,Phase 1
	TSK N,MKT,3/17/2022,3/19/2022,0.0,Phase 1
	TSK L,ENG,3/10/2022,3/13/2022,0.0,Phase 1
	TSK K,PROD,3/9/2022,3/13/2022,0.0,Phase 2
	TSK J,PROD,3/4/2022,3/17/2022,0.0,Phase 2


sample calls
gantter().gantt_demo()
gantter().gantt_From_csv('sampledata.csv','a title','Task','Stream','Start_dt','End_dt','Completionpct','%m/%d/%Y')


		"""
		print(notes)
	def prepare_graph_data(self):
		self.assignbars(self.df[self.taskgroup])

		# Using pandas.to_datetime() to convert pandas column to DateTime
		self.df[self.startfield] = pd.to_datetime(self.df[self.startfield], format="%m/%d/%Y")
		self.df[self.endfield] = pd.to_datetime(self.df[self.endfield], format="%m/%d/%Y")

		# project start date
		proj_start = self.df[self.startfield].min()

		# number of days from project start to task start
		self.df['start_num'] = (self.df[self.startfield]-proj_start).dt.days

		# number of days from project start to end of tasks
		self.df['end_num'] = (self.df[self.endfield]-proj_start).dt.days

		# days between start and end of each task
		self.df['days_start_to_end'] = self.df.end_num - self.df.start_num

		# days between start and current progression of each task
		self.df['current_num'] = (self.df.days_start_to_end * self.df[self.completionfield])

		self.df['color'] = self.df.apply(self.color, axis=1)

		return proj_start

	# create a column with the color for each team
	def color(self,row):
			return self.barcolors[row[self.taskgroup]]

	##### LEGENDS #####
	def build_legend(self):
		legend_elements = []
		for i in range(0,len(self.bars)):
			legend_elements.append(Patch(facecolor=self.barcolors[self.bars[i]], label=self.bars[i]))

		return legend_elements

	def check_fields_exist(self):
		missingfields = 0
		if self.tasknamefield not in self.df:
			print('No `' + self.tasknamefield + '` field in data')
			missingfields += 1

		if self.taskgroup not in self.df:
			print('No `' + self.taskgroup + '` field in data')
			missingfields += 1

		if self.startfield not in self.df:
			print('No `' + self.startfield + '` field in data')
			missingfields += 1

		if self.endfield not in self.df:
			print('No `' + self.endfield + '` field in data')
			missingfields += 1

		if self.completionfield not in self.df:
			print('No `' + self.completionfield + '` field in data')
			missingfields += 1


		if missingfields > 0:
			sys.exit(0)

	def gantt_demo(self):
		self.df = self.getdata_Demo()
		self.graphit()

	"""
		SELECT 'a Title' as title								-- field[0] = Title
				, 'Task A' as tasknamefield					-- field[1] = Task Name
				, 'Engineering Team' as taskgroup		-- field[2] = Task Group
				, '01/01/2023' as startfield				-- field[3] = Start Date
				, '02/02/2023' as endfield					-- field[4] = End Date
				, 0.5 as completionfield						-- field[5] = Completion percent from 0.0 - 1.0
	"""
	def gantt_From_sqlcsv(self,csvfilename,qry):
		self.schwiz.createload_sqlite_from_csv(csvfilename)
		tablename = self.schwiz.lastcall_tablename
		query = qry.replace(csvfilename,tablename)

		data = self.schwiz.dbthings.sqlite_db.query(query)
		self.schwiz.dbthings.sqlite_db.execute('DROP TABLE ' + tablename)
		
		cols = []
		for k in [i[0] for i in data.description]:
			cols.append(k)

		self.df = DataFrame(data)
		self.df.columns = cols
		Title = self.df['Title'].min()

		self.graphit(Title,cols[1],cols[2],cols[3],cols[4],cols[5])


	def gantt_From_csv(self,csvfilename,graphtitle='',tasknamefield='',taskgroup='',startfield='',endfield='',completionfield='',date_fmt='%m/%d/%Y'):

		self.df = self.getdata_fromcsvfile(csvfilename)

		self.graphit(graphtitle,tasknamefield,taskgroup,startfield,endfield,completionfield)
		
	def graphit(self,graphtitle='',tasknamefield='',taskgroup='',startfield='',endfield='',completionfield=''):

		if tasknamefield != '':
			self.tasknamefield = tasknamefield

		if taskgroup != '':
			self.taskgroup = taskgroup

		if startfield != '':
			self.startfield = startfield

		if endfield != '':
			self.endfield = endfield

		if completionfield != '':
			self.completionfield = completionfield

		self.check_fields_exist()

		#self.df = self.getdata_fromcsvfile('data.csv')
		#self.df = self.getdata_Demo()
		proj_start = self.prepare_graph_data()
		##### PLOT #####
		fig, (ax, ax1) = plt.subplots(2, figsize=(16,6), gridspec_kw={'height_ratios':[6, 1]})

		# bars
		ax.barh(self.df.Task, self.df.current_num, left=self.df.start_num, color=self.df.color)
		ax.barh(self.df.Task, self.df.days_start_to_end, left=self.df.start_num, color=self.df.color, alpha=0.5)

		for idx, row in self.df.iterrows():
				ax.text(row.end_num+0.1, idx, f"{int(row[self.completionfield]*100)}%", va='center', alpha=0.8)
				ax.text(row.start_num-0.1, idx, row.Task, va='center', ha='right', alpha=0.8)

		# grid lines
		ax.set_axisbelow(True)
		ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.2, which='both')

		# ticks
		xticks = np.arange(0, self.df.end_num.max()+1, 3)
		xticks_labels = pd.date_range(proj_start, end=self.df[self.endfield].max()).strftime("%m/%d")
		xticks_minor = np.arange(0, self.df.end_num.max()+1, 1)
		ax.set_xticks(xticks)
		ax.set_xticks(xticks_minor, minor=True)
		ax.set_xticklabels(xticks_labels[::3])
		ax.set_yticks([])

		# ticks top
		# create a new axis with the same y
		ax_top = ax.twiny()

		# align x axis
		ax.set_xlim(0, self.df.end_num.max())
		ax_top.set_xlim(0, self.df.end_num.max())

		# top ticks (markings)
		xticks_top_minor = np.arange(0, self.df.end_num.max()+1, 7)
		ax_top.set_xticks(xticks_top_minor, minor=True)
		# top ticks (label)
		xticks_top_major = np.arange(3.5, self.df.end_num.max()+1, 7)
		ax_top.set_xticks(xticks_top_major, minor=False)
		# week labels
		xticks_top_labels = [f"Week {i}"for i in np.arange(1, len(xticks_top_major)+1, 1)]
		ax_top.set_xticklabels(xticks_top_labels, ha='center', minor=False)

		# hide major tick (we only want the label)
		ax_top.tick_params(which='major', color='w')
		# increase minor ticks (to marks the weeks start and end)
		ax_top.tick_params(which='minor', length=8, color='k')

		# remove spines
		ax.spines['right'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.spines['left'].set_position(('outward', 10))
		ax.spines['top'].set_visible(False)

		ax_top.spines['right'].set_visible(False)
		ax_top.spines['left'].set_visible(False)
		ax_top.spines['top'].set_visible(False)

		plt.suptitle(graphtitle)
		
		legend_elements = self.build_legend()
		ax1.legend(handles=legend_elements, loc='upper center', ncol=5, frameon=False)

		# clean second axis
		ax1.spines['right'].set_visible(False)
		ax1.spines['left'].set_visible(False)
		ax1.spines['top'].set_visible(False)
		ax1.spines['bottom'].set_visible(False)
		ax1.set_xticks([])
		ax1.set_yticks([])

		plt.show()

	def assignbars(self,df_col):
		# assign colors
		i = 0
		for val in df_col.unique():
			if i >= len(self.colorlist[i]):
				i = 0
			self.bars.append(val)
			self.barcolors[val] = self.colorlist[i]
			i += 1

	def getdata_fromcsvfile(self,csvdatafile=''):
		return pd.read_csv(csvdatafile)

	def getdata_Demo(self):

		data = {self.tasknamefield: {0: 'TSK M',
										 1: 'TSK N',
										 2: 'TSK L',
										 3: 'TSK K',
										 4: 'TSK J',
										 5: 'TSK H',
										 6: 'TSK I',
										 7: 'TSK G',
										 8: 'TSK F',
										 9: 'TSK E',
										 10: 'TSK D',
										 11: 'TSK C',
										 12: 'TSK B',
										 13: 'TSK A'},

		self.taskgroup: {0: 'IT',
									1: 'MKT',
									2: 'ENG',
									3: 'PROD',
									4: 'PROD',
									5: 'FIN',
									6: 'MKT',
									7: 'FIN',
									8: 'MKT',
									9: 'ENG',
									10: 'FIN',
									11: 'IT',
									12: 'MKT',
									13: 'MKT'},
 
		self.startfield: {0: Timestamp('2022-03-17 00:00:00'),
						 1: Timestamp('2022-03-17 00:00:00'),
						 2: Timestamp('2022-03-10 00:00:00'),
						 3: Timestamp('2022-03-09 00:00:00'),
						 4: Timestamp('2022-03-04 00:00:00'),
						 5: Timestamp('2022-02-28 00:00:00'),
						 6: Timestamp('2022-02-28 00:00:00'),
						 7: Timestamp('2022-02-27 00:00:00'),
						 8: Timestamp('2022-02-26 00:00:00'),
						 9: Timestamp('2022-02-23 00:00:00'),
						 10: Timestamp('2022-02-22 00:00:00'),
						 11: Timestamp('2022-02-21 00:00:00'),
						 12: Timestamp('2022-02-19 00:00:00'),
						 13: Timestamp('2022-02-15 00:00:00')},
 
		self.endfield: {0: Timestamp('2022-03-20 00:00:00'),
					 1: Timestamp('2022-03-19 00:00:00'),
					 2: Timestamp('2022-03-13 00:00:00'),
					 3: Timestamp('2022-03-13 00:00:00'),
					 4: Timestamp('2022-03-17 00:00:00'),
					 5: Timestamp('2022-03-02 00:00:00'),
					 6: Timestamp('2022-03-05 00:00:00'),
					 7: Timestamp('2022-03-03 00:00:00'),
					 8: Timestamp('2022-02-27 00:00:00'),
					 9: Timestamp('2022-03-09 00:00:00'),
					 10: Timestamp('2022-03-01 00:00:00'),
					 11: Timestamp('2022-03-03 00:00:00'),
					 12: Timestamp('2022-02-24 00:00:00'),
					 13: Timestamp('2022-02-20 00:00:00')},
 
		self.completionfield: {0: 0.0,
									1: 0.0,
									2: 0.0,
									3: 0.0,
									4: 0.0,
									5: 1.0,
									6: 0.4,
									7: 0.7,
									8: 1.0,
									9: 0.5,
									10: 1.0,
									11: 0.9,
									12: 1.0,
									13: 1.0}}

		return pd.DataFrame(data)

if __name__ == '__main__':
	do_main()
