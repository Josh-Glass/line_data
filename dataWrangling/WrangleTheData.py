import data_wrangling_utils as utils
phase = {
    '1': ['study','typicality1','trainPhase','typicality2','similarity','testPhase'],
    '2': ['study','typicality1','trainPhase','typicality2','similarity','testPhase'],
    '3': ['trainPhase','typicality2','similarity','testPhase']}
path= 'C:/Users/apers/line_data/data'
#utils.changeFileType(path)

for key in phase:
    utils.scrape(path=path,phase= phase[key], cond= key)