"""
  Dave Skura
	https://towardsdatascience.com/gantt-charts-with-pythons-matplotlib-395b7af72d72

  
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from pandas import Timestamp

def do_main():
	#gantter().show_how_to() #
	gantter().graphit('Tasks by Department','Department')

class gantter():
	def __init__(self,gantdatacsvfilename = ''): # data.csv
		print(" chumbo ") #
		self.datafile = gantdatacsvfilename	
		self.colorlist = ['#E64646','#E69646', '#34D05C', '#34D0C3', '#3475D0','#336600','#663300','#990000','#3300CC','#CC0000']
		self.bars = []
		self.barcolors = {}
		self.tasknamefield = ''
		self.teamfield = ''
		self.startfield = ''
		self.endfield = ''
		self.completionfield = ''
		
	def show_how_to(self):
		notes = """
data expected to look as follows:

	,Task,Department,Start,End,Completion
	0,TSK M,IT,3/17/2022,3/20/2022,0.0
	1,TSK N,MKT,3/17/2022,3/19/2022,0.5
	...

		"""
		print(notes)
	def prepare_graph_data(self,df):
		self.assignbars(df[self.teamfield])

		# Using pandas.to_datetime() to convert pandas column to DateTime
		df[self.startfield] = pd.to_datetime(df[self.startfield], format="%m/%d/%Y")
		df[self.endfield] = pd.to_datetime(df[self.endfield], format="%m/%d/%Y")

		# project start date
		proj_start = df.Start.min()

		# number of days from project start to task start
		df['start_num'] = (df.Start-proj_start).dt.days

		# number of days from project start to end of tasks
		df['end_num'] = (df.End-proj_start).dt.days

		# days between start and end of each task
		df['days_start_to_end'] = df.end_num - df.start_num

		# days between start and current progression of each task
		df['current_num'] = (df.days_start_to_end * df[self.completionfield])

		df['color'] = df.apply(self.color, axis=1)

		return df,proj_start

	# create a column with the color for each team
	def color(self,row):
			return self.barcolors[row[self.teamfield]]

	##### LEGENDS #####
	def build_legend(self):
		legend_elements = []
		for i in range(0,len(self.bars)):
			legend_elements.append(Patch(facecolor=self.barcolors[self.bars[i]], label=self.bars[i]))

		return legend_elements

	def graphit(self
						,graphtitle				='PROJECT X'
						,tasknamefield		='Task'
						,teamfield				='Department'
						,startfield				='Start'
						,endfield					='End'
						,completionfield	='Completion'):

		self.tasknamefield = tasknamefield
		self.teamfield = teamfield
		self.startfield = startfield
		self.endfield = endfield
		self.completionfield = completionfield

		df = self.getdata_fromcsvfile('data.csv')
		#df = self.getdata_Demo()
		df,proj_start = self.prepare_graph_data(df)

		##### PLOT #####
		fig, (ax, ax1) = plt.subplots(2, figsize=(16,6), gridspec_kw={'height_ratios':[6, 1]})

		# bars
		ax.barh(df.Task, df.current_num, left=df.start_num, color=df.color)
		ax.barh(df.Task, df.days_start_to_end, left=df.start_num, color=df.color, alpha=0.5)

		for idx, row in df.iterrows():
				ax.text(row.end_num+0.1, idx, f"{int(row[self.completionfield]*100)}%", va='center', alpha=0.8)
				ax.text(row.start_num-0.1, idx, row.Task, va='center', ha='right', alpha=0.8)

		# grid lines
		ax.set_axisbelow(True)
		ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.2, which='both')

		# ticks
		xticks = np.arange(0, df.end_num.max()+1, 3)
		xticks_labels = pd.date_range(proj_start, end=df.End.max()).strftime("%m/%d")
		xticks_minor = np.arange(0, df.end_num.max()+1, 1)
		ax.set_xticks(xticks)
		ax.set_xticks(xticks_minor, minor=True)
		ax.set_xticklabels(xticks_labels[::3])
		ax.set_yticks([])

		# ticks top
		# create a new axis with the same y
		ax_top = ax.twiny()

		# align x axis
		ax.set_xlim(0, df.end_num.max())
		ax_top.set_xlim(0, df.end_num.max())

		# top ticks (markings)
		xticks_top_minor = np.arange(0, df.end_num.max()+1, 7)
		ax_top.set_xticks(xticks_top_minor, minor=True)
		# top ticks (label)
		xticks_top_major = np.arange(3.5, df.end_num.max()+1, 7)
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

		self.teamfield: {0: 'IT',
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
