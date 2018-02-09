import shutil
import os

batchName = "150305"

os.chdir(batchName)

if not os.path.exists("all"):
        os.makedirs("all")

dirNames = [d for d in os.listdir() if (os.path.isdir(os.path.join(d)) and d != "all")]
dirNames.sort()



index = 0

for d in dirNames:
#    fileNames = [f for f in os.listdir(os.path.join(d)) if (os.path.isfile(os.path.join(f)) and ".jpg" in f)]
    fileNames = [f for f in os.listdir(os.path.join(d))]
    fileNames.sort()
    for f in fileNames:
        shutil.copy(d + '/' + f, "all")
        newFileName = "{0}_{1}.jpg".format(batchName, index)
        os.rename('all/' + f, 'all/' + newFileName)
        index += 1
print("done")