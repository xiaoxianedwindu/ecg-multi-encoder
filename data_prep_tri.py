import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from utils import *
from config import get_config

#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

config = get_config()

(X,y, group) = loaddata_nosplit_scaled_index(config.input_size, config.feature)
classes = ['A', 'E', 'j', 'L', 'N', 'P', 'R', 'V']
#Xe = np.expand_dims(X, axis=2)

y_new = np.column_stack((np.array(y), np.array(group)))

from sklearn.model_selection import train_test_split
X, Xval, y, yval = train_test_split(X, y_new, test_size=0.25, random_state=1)

group_new= np.split(y,[8,9], axis=1)[1]
y = np.split(y,[8,9], axis=1)[0]
y = np.array(pd.DataFrame(y).apply(pd.to_numeric))          #https://stackoverflow.com/a/34844867

yval_subjectlabel= np.split(yval,[8,9], axis=1)[1]
yval = np.split(yval,[8,9], axis=1)[0]
yval = np.array(pd.DataFrame(yval).apply(pd.to_numeric))

subject = np.unique(group_new)#, return_counts = True)
beat_table = pd.DataFrame()
for s in subject:
    beat_table = pd.concat([beat_table, pd.DataFrame(y[np.where(group_new == s)[0]], columns = classes).apply(pd.to_numeric).sum()], axis=1)
beat_table = beat_table.T
beat_table.index = subject

val_subject = np.unique(yval_subjectlabel)#, return_counts = True)
val_beat_table = pd.DataFrame()
for s in val_subject:
    val_beat_table = pd.concat([val_beat_table, pd.DataFrame(yval[np.where(yval_subjectlabel == s)[0]], columns = classes).apply(pd.to_numeric).sum()], axis=1)
val_beat_table = val_beat_table.T
val_beat_table.index = val_subject

#selected_beat_type = 'j'
#selected_subject = '207'

print(beat_table)
print(val_beat_table)
'''
beat_index_subject = np.array(beat_table.where(beat_table[selected_beat_type]>0).dropna(how='all').index)        #array of subjects with selected beat type
print(beat_index_subject)                               #subjects that have selected beat type
print(beat_table.where(beat_table[selected_beat_type]>0).sum())        #sum of beat count of subjects with selected type by beat type
print(beat_table.where(beat_table[selected_beat_type]>0).sum().sum())  #total sum of beat count of subjects with selected type
'''
print("=========")
print(X.shape)                                          #original total beat count

X_subjects_total = []
y_subjects_total = []
X_beattype_total = []
y_beattype_total = []
X_subjects_beat_total = []
y_subjects_beat_total = []
X_trip_p_total = []
X_trip_s_total = []
X_trip_r_total = []
for selected_beat_type in tqdm(classes):
#for selected_beat_type in tqdm(['A','E']):
    beat_index_subject = np.array(beat_table.where(beat_table[selected_beat_type]>0).dropna(how='all').index)        #array of subjects with selected beat type
    print(beat_index_subject)                                               #subjects that have selected beat type
    print(beat_table.where(beat_table[selected_beat_type]>0).sum())         #sum of beat count of subjects with selected type by beat type
    print(beat_table.where(beat_table[selected_beat_type]>0).sum().sum())   #total sum of beat count of subjects with selected type
    for selected_subject in tqdm(beat_index_subject):
        X_subjects = []
        y_subjects = []
        X_beattype = []
        y_beattype = []
        X_subjects_beat = []
        y_subjects_beat = []
        X_trip_p = []
        X_trip_s = []
        X_trip_r = []
        for x in range(X.shape[0]):
            if group_new[x] == selected_subject:    #collects beats from selected subject
                X_subjects.append(X[x])
                y_subjects.append(y[x])

            if group_new[x] in beat_index_subject and y[x][classes.index(selected_beat_type)] == 1: #collects beats of the selected beat type from subjects that have the selected beat type
                X_beattype.append(X[x])
                y_beattype.append(y[x])

            if group_new[x] == selected_subject and y[x][classes.index(selected_beat_type)] == 1:   #collects beats of the selected beat type from the selected subject
                X_subjects_beat.append(X[x])
                y_subjects_beat.append(y[x])

            if group_new[x] in beat_index_subject and group_new[x] != selected_subject and y[x][classes.index(selected_beat_type)] == 1:    #collects beats not of the selected beat type from other subjects that also have the selected beat type
                X_trip_p.append(X[x])

            if group_new[x] == selected_subject and y[x][classes.index(selected_beat_type)] != 1:   #collects beat not of the selected beat type from the selected subject
                X_trip_s.append(X[x])

            if len(X_subjects_beat) > 500 and len(X_trip_s) > 0 or len(X_beattype) > 3000:  #undersample majority beat class types by stopping data collection
                break
        if len(X_trip_s) == 0:
            X_trip_s = X_subjects_beat      #in case the selected subject has no other beats other than the selected beat type

        selected_subject_count = len(X_subjects)
        beattype_count = len(X_beattype) 
        subject_beat_count = len(X_subjects_beat)
        print(selected_subject_count, beattype_count, subject_beat_count)
        trip_p_count = len(X_trip_p)
        trip_s_count = len(X_trip_s)
        print(trip_p_count, trip_s_count)
        print("=========================")
        #to ensure the data is of the same size, random resampling of the beat collections is used to match the largest collection
        import random
        def random_oversample(data, label, count):
            r = random.randint(0,count-1)
            data.append(data[r])
            label.append(label[r])
            return data, label

        def random_oversample_1(data, count):
            r = random.randint(0,count-1)
            data.append(data[r])
            return data

        top_count = selected_subject_count if selected_subject_count > beattype_count else beattype_count       #sets the size of the largest collection as target count
        #if collection is smaller than top_count, randomly oversample existing collections until all collection sizes match
        if selected_subject_count > beattype_count:
            while selected_subject_count > beattype_count:
                X_beattype, y_beattype = random_oversample(X_beattype, y_beattype, beattype_count)
                beattype_count += 1
        else:
            while beattype_count > selected_subject_count:
                X_subjects, y_subjects = random_oversample(X_subjects, y_subjects, selected_subject_count)
                selected_subject_count += 1

        while top_count > subject_beat_count:
            X_subjects_beat, y_subjects_beat = random_oversample(X_subjects_beat, y_subjects_beat, subject_beat_count)
            subject_beat_count += 1

        if trip_p_count > top_count:
            X_trip_p_temp = []
            X_trip_p_temp_count = 0
            while top_count > X_trip_p_temp_count:
                X_trip_p_temp = random_oversample_1(X_trip_p, trip_p_count)
                X_trip_p_temp_count += 1
            X_trip_p = X_trip_p_temp

        while top_count > trip_p_count:
            X_trip_p = random_oversample_1(X_trip_p, trip_p_count)
            trip_p_count += 1

        if trip_s_count > top_count:
            X_trip_s_temp = []
            X_trip_s_temp_count = 0
            while top_count > X_trip_s_temp_count:
                X_trip_s_temp = random_oversample_1(X_trip_p, trip_p_count)
                X_trip_s_temp_count += 1
            X_trip_s_temp = X_trip_s_temp

        while top_count > trip_s_count:
            X_trip_s = random_oversample_1(X_trip_s, trip_s_count)
            trip_s_count += 1

        for x in range(X.shape[0]):
            if group[x] != selected_subject and y[x][classes.index(selected_beat_type)] != 1:
                r = random.randint(0,1)
                if (r == 1): X_trip_r.append(X[x]) 
            if np.array(X_trip_r).shape[0] >= top_count:
                break
        trip_ref_count = np.array(X_trip_r).shape[0]

        print(top_count)

        #collect beat types from this loop instance
        def collect_values(collector, data):
            return collector + data

        X_subjects_total = collect_values(X_subjects_total, X_subjects)
        y_subjects_total = collect_values(y_subjects_total, y_subjects)
        X_beattype_total = collect_values(X_beattype_total, X_beattype)
        y_beattype_total = collect_values(y_beattype_total, y_beattype)
        X_subjects_beat_total = collect_values(X_subjects_beat_total, X_subjects_beat)
        y_subjects_beat_total = collect_values(y_subjects_beat_total, y_subjects_beat)
        X_trip_p_total = collect_values(X_trip_p_total, X_trip_p)
        X_trip_s_total = collect_values(X_trip_s_total, X_trip_s)
        X_trip_r_total = collect_values(X_trip_r_total, X_trip_r)


X_test = np.array([np.array(X_beattype_total), np.array(X_subjects_total), np.array(X_subjects_beat_total), np.array(X_trip_p_total), np.array(X_trip_s_total), np.array(X_trip_r_total)])
y_test = np.array([np.array(y_beattype_total), np.array(y_subjects_total), np.array(y_subjects_beat_total)])
print(X_test.shape, y_test.shape)
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2], X_test.shape[3])
#y_test = y_test.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2], y_test.shape[3])
#print(X_test.shape, y_test.shape)
print("=========================")


X_test_temp = []
for x in X_test:
  X_test_temp.append(np.expand_dims(x, axis=2))
X_test = X_test_temp
'''
y_test_temp = []
for y in y_test:
  y_test_temp.append(np.array(pd.DataFrame(y).idxmax(axis=1)))
y_test = y_test_temp
'''


# Define and save data

input_train = X_test
target_train = y_test

Xvale = np.expand_dims(Xval, axis=2)
yvale =yval.reshape(yval.shape[0], 1, yval.shape[1])

input_test = Xvale
target_test = yvale

import deepdish as dd
dd.io.save('dataset/traindata_tri.hdf5', input_train)
dd.io.save('dataset/trainlabel_tri.hdf5', target_train)
dd.io.save('dataset/testdata_tri.hdf5', input_test)
dd.io.save('dataset/testlabel_tri.hdf5', target_test)