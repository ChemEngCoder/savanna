from os.path import join

path = join("..", "..", "scratch", "evo", "pretraining_or_both", "gtdb_v220_imgpr")
files = [f for f in listdir(path)]
print(files)