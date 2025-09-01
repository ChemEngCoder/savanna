from os.path import join, listdir

path = join("..", "..", "scratch", "evo", "pretraining_or_both", "gtdb_v220_imgpr")
files = [f for f in listdir(path)]
print(files)