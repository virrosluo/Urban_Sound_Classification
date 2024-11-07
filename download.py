import soundata

dataset = soundata.initialize('urbansound8k', data_home="./data")
dataset.download()  # download the dataset
dataset.validate()  # validate that all the expected files are there

example_clip = dataset.choice_clip()  # choose a random example clip
print(example_clip)  # see the available data
