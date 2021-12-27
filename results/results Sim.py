import pandas as pd
import matplotlib.pyplot as plt


title = "Cumulative matching characteristic curve"
ylabel = "Accuracy"


plt.plot([1,2,3,4,5,6,7],[ 0.7368421052631579,0.8421052631578947,0.8947368421052632,0.9210526315789473,0.9473684210526315,0.9736842105263158,1],  marker='o')

#plt.legend(losses, loc='upper left')
plt.title(title)
plt.axis([0.5,7.5,0.5,1.05])
plt.ylabel(ylabel)
plt.xlabel("Rank score")
plt.show()
