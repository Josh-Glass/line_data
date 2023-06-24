import analysis_utils as au

learn_curve_paths = {
    'C:/Users/apers/line_data/cond_1_trainPhase_results.csv' : 'Learning Curve Reduced Classification',
    'C:/Users/apers/line_data/cond_2_trainPhase_results.csv' : 'Learning Curve Full Classsification',
    'C:/Users/apers/line_data/cond_3_trainPhase_results.csv' : 'Learning Curve No Observation',
}
gen_paths = {
    'C:/Users/apers/line_data/cond_1_testPhase_results.csv' : 'Generalization Reduced Classification',
    #'C:/Users/apers/line_data/cond_2_testPhase_results.csv' : 'Generalization Full Classification',
    #'C:/Users/apers/line_data/cond_3_testPhase_results.csv' : 'Generalization No Observation',
}
typ_paths = {
    'C:/Users/apers/line_data/cond_1_typicality1_results.csv' : 'Typ1 Reduced Classification',
    'C:/Users/apers/line_data/cond_1_typicality2_results.csv' : 'Typ2 Reduced Classification',
    'C:/Users/apers/line_data/cond_2_typicality1_results.csv' : 'Typ1 Full Classification',
    'C:/Users/apers/line_data/cond_2_typicality2_results.csv' : 'Typ2 Full Classification',
    'C:/Users/apers/line_data/cond_3_typicality2_results.csv' : 'No Observation',

}

changes= {
    'A50': 'A1',
    'A150': 'A2',
    'A250': 'A3',
    'A350': 'A4',
    'A450': 'A5',

    'B550': 'B1',
    'B650': 'B2',
    'B750': 'B3',
    'B850': 'B4',
    'B950': 'B5',

    'C1050': 'C1',
    'C1150': 'C2',
    'C1250': 'C3',
    'C1350': 'C4',
    'C1450': 'C5',
}


for key in learn_curve_paths:
    au.plotLearnCurve(path= key, title= learn_curve_paths[key])

for key in typ_paths:
   au.plotTyps(path=key, title= typ_paths[key], changestimID=True,changes=changes)



#for key in gen_paths:
    #au.getgens(path=key, title=gen_paths[key])
