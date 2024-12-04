import imageio.v3 as imageio
import os

#writer = imageio.get_writer("stress_free_1_1_1e7_A.gif", mode="I")

frames_path = 'frames/'
fnames = [frames_path + val for val in sorted(os.listdir(path=frames_path))][7000:]
#print(*fnames[:10] , sep='\n')

# for i, filename in enumerate(fnames):
# 	print(f"frame {i+1}/{len(fnames)} \t {filename}", end="\r")
# 	image = imageio.imread(filename)
# 	writer.append_data(image)
# print('')

images = []
for i, filename in enumerate(fnames):
	images.append(imageio.imread(filename))
	print(f"Added frame {i+1}/{len(fnames)} \t {filename}", end="\r")

#print(*images[:10] , sep='\n')

print('\nFiles read. Merging images.')
imageio.imwrite("stress_free_1_1_1e7_toend.mp4", images)