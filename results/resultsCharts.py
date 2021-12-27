import pandas as pd
import matplotlib.pyplot as plt

csv_name = 'logs/log_500.txt'
losses = ["discrim_loss", "gen_loss_GAN", "gen_loss_L1"]

title = "Losses"
ylabel = "Losses"

data = pd.read_csv(csv_name, sep=' ; ', engine='python')

print(data.loc[:, 'discrim_loss'].values)
#number_people = data.loc["number_people", :].values
for loss in losses:
    plt.plot(data.loc[:, loss].values)

plt.legend(losses, loc='upper left')
plt.title(title)
plt.ylabel(ylabel)
plt.xlabel("Steps")
plt.show()
