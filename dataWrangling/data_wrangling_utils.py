import os
import json
import csv
import pandas as pd
import numpy as np



def changeFileType(path):
    # Folder Path
    path = str(path)
    
    # Change the directory
    os.chdir(path)
    
    # Read text File
    
        # iterate through all file
    i = 1
    for file in os.listdir(path=path):
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}/{file}"
            
            with open(file, "r") as data:
                # remove any old .json files
                #os.remove(f"{path}/{file}.json")
                # write .txt files to .json line by line
                with open(f"{path}/{i}.json", "w") as output:
                    for line in data.readlines():
                        output.write(line)
        i += 1




def checkCrit(subject):
    data = np.array(subject['results']['trainPhase'][1:])
    accData= data[:,5]
    accData[accData == 'True'] = int(1)
    accData[accData == 'False'] = int(0)
    accData = accData.astype(int)
    end= 1
    start= 0
    windowAcc = []
    while end < len(accData):
        if np.mean(accData[start:end]) == 1:
            windowAcc.append(1)
        else:
            windowAcc.append(0)
        end +=1
        start +=1
    print(np.sum(windowAcc))
    if np.sum(windowAcc)>=1:
        metcrit = 1
    else:
        metcrit = 0
    return metcrit





def scrape(path, phase, cond):
    path = str(path)


    for i in phase:
        alldata = []
        
        for file in os.listdir(path):
            if file.endswith('.json'):
            
                with open(os.path.join(path, file), 'r') as filedata:
                    subject = json.load(filedata)
                    if checkCrit(subject=subject) == 1:
                        if subject['condition'] == str(cond):
                    

                        
                    
                            
                            
                            
                            ## get  data
                            subjData = pd.DataFrame(
                                subject['results'][i][1:],
                                columns = subject['results'][i][0],
                                )
                            subjData['condition'] = subject['condition']
                            subjData['id'] = subject['id']
                            subjData['phase'] = i

                                
                                

                            subjDataConcat = pd.concat([
                                subjData, 
                            ], ignore_index = True)

                            alldata.append(subjDataConcat)
                            

        alldata = pd.concat(alldata, ignore_index = True) 
        alldata.to_csv('cond_'+str(cond) + '_' + str(i) + '_results.csv', index = None)






