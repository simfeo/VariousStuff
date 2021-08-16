# -*- coding: utf-8 -*-
import os
import datetime

dct = {}

for el in filter (lambda x: x.lower().endswith(".jpg") or x.lower().endswith(".nef"), os.listdir(u'.')):
	nn = str(datetime.datetime.fromtimestamp(os.path.getmtime(el)))
	nn = nn.replace(":","").replace("-","").replace(" ","_")
	if nn in dct:
		if type (dct[nn]) != list:
			dct[nn]= [dct[nn]]
		dct[nn].append(el)
	else:
		dct[nn] = el

# print dct.keys()

# for k in dct.keys():
# 	print k, dct[k]

def rename_file(file_name, new_name, ind):
	if ind == 0:
		os.rename(file_name, file_name.replace(file_name[:-4], new_name))
	else:
		os.rename(file_name, file_name.replace(file_name[:-4], new_name+"_"+str(ind)))

for k in dct.keys():
	if type (dct[k]) != list:
		file_name = dct[k]
		rename_file(file_name,k,0)
	else:
		for i in range(len(dct[k])):
			file_name = dct[k][i]
			rename_file(file_name,k,i)				
