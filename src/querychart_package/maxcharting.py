"""
  Dave Skura
  
"""
import numpy as np
import matplotlib.pyplot as plt


title = 'What is this graph telling you?'
# creating the dataset
data = {'C': 20, 'C++': 15, 'Java': 30,
		'Python': 35}

courses = ['A','B','C']
values = [10,20,30]

fig = plt.figure(figsize=(10, 5)) # Width x Height in inches

# creating the bar plot
plt.barh(courses, values, color='maroon',)

plt.xlabel("No. of students enrolled")
plt.ylabel("Courses offered")
plt.title(title)
plt.show()

