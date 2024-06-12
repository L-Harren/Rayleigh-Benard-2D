import h5py
import matplotlib.pyplot as plt
import numpy as np

#folder = r'### PROJECT/RB_Nusselt/snapshots_1e5/'
folder = r'snapshots/'
data = []
for i in range(1,8): 
	file = folder + 'snapshots_s' + str(i) + '.h5'
	with h5py.File(file, 'r') as hf:
		for val in hf['tasks']['Nusselt']:
			data.append( val[:, 0][0] )

#print(data[141:])
Nu = np.mean(data[len(data)//2:])

plt.figure(figsize=(12,5))
plt.title(f'Mean Nu = {float(Nu)}')
plt.xlabel('Time (time scale units)', fontsize=14)
plt.ylabel('Nusselt' , fontsize=14)
#plt.yticks([])
plt.scatter(np.arange(len(data))/4 , data , s=12 , c = 'darkorange')
#plt.plot([35,62] , [Nu,Nu])
plt.tight_layout()
plt.show()