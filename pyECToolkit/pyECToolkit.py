import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
from io import StringIO
from scipy.interpolate import interp1d
import os
from lxml import html
import re
import requests
import json
from io import StringIO

#Note: This disables any warnings raised by requests.
requests.packages.urllib3.disable_warnings()


#### PARSERS FOR VARIOUS FILE TYPES ####

def parse_IV(file_or_folder,ref_potential=0.239):
    """
    A general parser for IV Curves (local), folders or files
    """
    if os.path.isfile(file_or_folder): #Parse as a file
        files_li = [file_or_folder]
    else: #Parse as a folder
        files_li = [file_or_folder+'/'+i for i in \
                    os.listdir(file_or_folder) if i!='.DS_Store']
    data_dfs = []
    for i in files_li:
        t = open(i).read()
        data = t[t.index('Potential/V'):].split('\n')
        columns = data[0].split('\t')[:2]
        if columns[1] == 'Current/A':
            continue
        float_data = []
        for j in data[2:-1]:
            new_pt = [float(k) for k in j.split('\t')][:2]
            new_pt[0]+=ref_potential
            float_data.append(new_pt)
        df = pd.DataFrame(float_data,columns=columns)
        data_dfs.append(df)
    return data_dfs, files_li

def parse_cycloV(file_or_folder,ref_potential=0.239):
    if os.path.isfile(file_or_folder): #Parse as a file
        files_li = [file_or_folder]
    else: #Parse as a folder
        files_li = [file_or_folder+'/'+i for i in \
                    os.listdir(file_or_folder) if i!='.DS_Store']
    data_dfs = []
    for i in files_li:
        t = open(i).read()
        data = t[t.index('Potential/V'):].split('\n')
        columns = data[0].split('\t')[:2]
        if columns[1] == 'i1/A':
            continue
        float_data = []
        for j in data[2:-1]:
            new_pt = [float(k) for k in j.split('\t')][:2]
            new_pt[0]+=ref_potential
            float_data.append(new_pt)
        df = pd.DataFrame(float_data,columns=columns)
        data_dfs.append(df)
    return data_dfs, files_li

def parse_XRD():
    return

#### RUNNING VARIOUS ALGORITHMS ON PARSED DATA ####

def IV_compute_half(test_1,plot_show=False,savefile=None):
    avg_ic_x = sorted(list(set(test_1['Potential/V'])))
    avg_ic_y = []
    for i in avg_ic_x:
        avg_ic_y.append(np.average(test_1.loc[test_1['Potential/V'] == i]['i1/A'].values))
    deriv = np.diff([avg_ic_y,avg_ic_x])[0]/np.diff([avg_ic_y,avg_ic_x][1])
    deriv2 = np.diff([deriv,avg_ic_x[1:]])[0]/np.diff([deriv,avg_ic_x[1:]][1])
    lower_split_x = avg_ic_x[:np.argmax(deriv2)-8]
    lower_split_y = avg_ic_y[:len(lower_split_x)]
    
    lower_m = 0
    lower_int = np.average(lower_split_y)
    
    upper_split_x = avg_ic_x[-3:]
    upper_split_y = avg_ic_y[-3:]
    
    upper_m = 0
    upper_int = 0
    
    middle_m = lower_m
    middle_int = np.average([lower_int,upper_int])
    mid_line = np.array([middle_m*i+middle_int for i in avg_ic_x])

    idx = np.argwhere(np.diff(np.sign(mid_line - avg_ic_y))).flatten()[0]
    pt1_1 = [avg_ic_x[idx],avg_ic_y[idx]]
    pt1_2 = [avg_ic_x[idx+1],avg_ic_y[idx+1]]
    pt2_1 = [avg_ic_x[idx],mid_line[idx]]
    pt2_2 = [avg_ic_x[idx+1],mid_line[idx+1]]
    m1 = (pt1_1[1]-pt1_2[1])/(pt1_1[0]-pt1_2[0])
    m2 = (pt2_1[1]-pt2_2[1])/(pt2_1[0]-pt2_2[0])
    b1 = pt1_1[1]-pt1_1[0]*m1
    b2 = pt2_1[1]-pt2_1[0]*m2

    half_way = (b2-b1)/(m1-m2)
    half_way_current = half_way*middle_m+middle_int
    if plot_show:
        plt.figure(figsize=(8,6))
        plt.plot(avg_ic_x,avg_ic_y,label='IC Curve (Avg)')
        plt.plot(avg_ic_x,[upper_m*i+upper_int for i in avg_ic_x],label='Upper Fit')
        plt.plot(avg_ic_x,[lower_m*i+lower_int for i in avg_ic_x],label='Lower Fit')
        plt.plot(avg_ic_x,mid_line,label='Middle Path')
        plt.scatter([half_way],[half_way_current],color='black',s=50,label='Half Way')
        plt.text(half_way,half_way_current,\
                 'Half Way Potential: '+str(round(half_way,4))+' ',fontsize=14,\
                 color='black',horizontalalignment='right')        
        plt.grid()
        plt.title('IV Curve',fontsize=14)
        plt.xlabel('Potential/V')
        plt.ylabel('i1/A')
        plt.legend()
        plt.show()
        plt.close()
    if savefile is not None:
        plt.figure(figsize=(8,6))
        plt.plot(avg_ic_x,avg_ic_y,label='IC Curve (Avg)')
        plt.plot(avg_ic_x,[upper_m*i+upper_int for i in avg_ic_x],label='Upper Fit')
        plt.plot(avg_ic_x,[lower_m*i+lower_int for i in avg_ic_x],label='Lower Fit')
        plt.plot(avg_ic_x,mid_line,label='Middle Path')
        plt.scatter([half_way],[half_way_current],color='black',s=50,label='Half Way')
        plt.text(half_way,half_way_current,\
                 'Half Way Potential: '+str(round(half_way,4))+' ',fontsize=14,\
                 color='black',horizontalalignment='right')        
        plt.grid()
        plt.title('IV Curve',fontsize=14)
        plt.xlabel('Potential/V')
        plt.ylabel('i1/A')
        plt.legend()
        plt.savefig(savefile)
        plt.close()
    return half_way, half_way_current

def IV_get_potential(df,potential):
    avg_ic_x = sorted(list(set(df['Potential/V'])))
    avg_ic_y = []
    for i in avg_ic_x:
        avg_ic_y.append(np.average(df.loc[df['Potential/V'] == i]['i1/A'].values))
    
    find_pt = interp1d(avg_ic_x,avg_ic_y)
    c = float(find_pt(potential))
    return c


def cycloV_compute_area(test_2,upper=0.5,lower=0.1,plot_show=False,savefile=None):
    pot_mins = list(np.where(test_2['Potential/V']==test_2['Potential/V'].min())[0])
    pot_maxs = list(np.where(test_2['Potential/V']==test_2['Potential/V'].max())[0])

    descending_potential = []
    descending_current = []
    ascending_potential = []
    ascending_current = []

    if pot_maxs[0]<pot_mins[0]: #Starts at the highest potential, goes down
        for i in range(len(pot_maxs)):
            descending_potential.append(test_2['Potential/V'][pot_maxs[i]:pot_mins[i]+1])
            descending_current.append(test_2['Current/A'][pot_maxs[i]:pot_mins[i]+1])
        for i in range(len(pot_maxs)-1):
            ascending_potential.append(test_2['Potential/V'][pot_mins[i]:pot_maxs[i+1]+1])
            ascending_current.append(test_2['Current/A'][pot_mins[i]:pot_maxs[i+1]+1])
    else: #Starts at the lowest potential, goes up
        for i in range(len(pot_maxs)-1):
            descending_potential.append(test_2['Potential/V'][pot_maxs[i]:pot_mins[i+1]+1])
            descending_current.append(test_2['Current/A'][pot_maxs[i]:pot_mins[i+1]+1])
        for i in range(len(pot_maxs)):
            ascending_potential.append(test_2['Potential/V'][pot_mins[i]:pot_maxs[i]+1])
            ascending_current.append(test_2['Current/A'][pot_mins[i]:pot_maxs[i]+1])


    line1_x = np.average(descending_potential,axis=0)
    line1_y = np.average(descending_current,axis=0)
    line2_x = np.average(ascending_potential,axis=0)
    line2_y = np.average(ascending_current,axis=0)

    x_coords = np.round(line1_x,3)
    if np.average(line1_y)>np.average(line2_y):
        y_upper = line1_y
        y_lower = np.flip(line2_y)
    else:
        y_lower = line1_y
        y_upper = np.flip(line2_y)


    ub = np.where(x_coords==lower)[0][0]
    lb = np.where(x_coords==upper)[0][0]
    shortened_lower = y_lower[lb:ub]
    shortened_upper = y_upper[lb:ub]
    shortened_x = x_coords[lb:ub]

    total_area = abs(np.trapz(shortened_upper,x=shortened_x))+\
                 abs(np.trapz(shortened_lower,x=shortened_x))
    h_val = total_area/abs(lower-upper)
    if plot_show:
        plt.figure(figsize=(8,6))
        plt.plot(test_2['Potential/V'],test_2['Current/A'],color='gray',\
                 alpha=0.6,label='Original Cycles')
        plt.plot(x_coords,y_lower,label='Avg. Lower Cycle')
        plt.plot(x_coords,y_upper,label='Avg. Upper Cycle')
        plt.fill_between(shortened_x, shortened_lower, shortened_upper,\
                         color='lightgreen',alpha=0.6,label='Test Region')
        plt.legend()
        plt.grid()
        plt.xlabel('Potential/V')
        plt.ylabel('Current/A')
        plt.title('Cyclovoltammetry',fontsize=14)

        plt.text(np.average(shortened_x),0,\
                 'Total Area: '+str(round(total_area,8)),fontsize=12,\
                 color='darkgreen',horizontalalignment='center')
        plt.text(np.average(shortened_x),-0.0005,\
                 'Area/Width: '+str(round(h_val,8)),fontsize=12,\
                 color='darkgreen',horizontalalignment='center')
        plt.show()
        plt.close()
    if savefile is not None:
        plt.figure(figsize=(8,6))
        plt.plot(test_2['Potential/V'],test_2['Current/A'],color='gray',\
                 alpha=0.6,label='Original Cycles')
        plt.plot(x_coords,y_lower,label='Avg. Lower Cycle')
        plt.plot(x_coords,y_upper,label='Avg. Upper Cycle')
        plt.fill_between(shortened_x, shortened_lower, shortened_upper,\
                         color='lightgreen',alpha=0.6,label='Test Region')
        plt.legend()
        plt.grid()
        plt.xlabel('Potential/V')
        plt.ylabel('Current/A')
        plt.title('Cyclovoltammetry',fontsize=14)

        plt.text(np.average(shortened_x),0,\
                 'Total Area: '+str(round(total_area,8)),fontsize=12,\
                 color='darkgreen',horizontalalignment='center')
        plt.text(np.average(shortened_x),-0.0005,\
                 'Area/Width: '+str(round(h_val,8)),fontsize=12,\
                 color='darkgreen',horizontalalignment='center')
        plt.savefig(savefile)
        plt.close()
    return total_area, h_val

def unmix_xrd():
    return


#### ELECTROCAT INTERACTIONS ####


def EC_extract(private_api_key,dataset_name,resource_name=None,dev=False):
    apiheader={ 'Authorization':str(private_api_key)}
    if dev:
        datahub = 'dev-datahub.electrocat.org'
    else:
        datahub = 'datahub.electrocat.org'
    getfile = requests.get('https://'+datahub+'/dataset/'+dataset_name.lower(),\
                           headers=apiheader,verify=False)
    if getfile.ok:
        filedata=getfile.content
    else:
        raise ValueError('Bad Link')
    page = getfile.content
    tree = html.fromstring(filedata)
    hrefs = tree.xpath('//a/@href')
    if resource_name is not None: #Parse for a single file
        matches = [i for i in hrefs if resource_name in i]
        if len(matches) == 0:
            raise KeyError('No valid mathces found for given resource name')
        link = matches[0]
        dataset_id = re.findall('dataset/.*/resource',\
                                  link)[0].replace('dataset/','').replace('/resource','')
        resource_id = re.findall('resource/.*/download',\
                                   link)[0].replace('resource/','').replace('/download','')
        return dataset_id, resource_id
    else:
        dataset_ids = []
        resource_ids = []
        all_resources = [i for i in hrefs if '/resource/' in i and '/download' in i]
        for link in all_resources:
            dataset_ids.append(re.findall('dataset/.*/resource',\
                                      link)[0].replace('dataset/','').replace('/resource',''))
            resource_ids.append(re.findall('resource/.*/download',\
                                       link)[0].replace('resource/','').replace('/download',''))
        return dataset_ids, resource_ids

def EC_upload_file(private_api_key, dataset_name, file_path, description=None,dev=False):
    if description is None:
        description = ''
    if dev:
        datahub = 'dev-datahub.electrocat.org'
    else:
        datahub = 'datahub.electrocat.org'
    dataset_ids, _ = EC_extract(private_api_key,dataset_name)
    file_name = os.path.split(file_path)[1]
    resource_metadata = {
       'package_id': dataset_ids[0] ,
       'name': file_name ,
       'description': description
       }
    
    apiurl='https://'+datahub+'/api/3/action/'
    apiheader={ 'Authorization':str(private_api_key)}
    
    r = requests.post(apiurl+'resource_create',
            data=resource_metadata,
            headers=apiheader,
            files=[('upload', open(file_path, 'rb'))])
    
    if r.status_code!=200:
        print('ERROR: Error uploading file '+file_name)
        return
    else:
        print('Successful upload to dataset '+dataset_name+\
              '. Resource ID: '+str(r.json()['result']['id']))

def EC_upload_folder(private_api_key, dataset_name, folder_path, descriptions=None,dev=False):
    files_present = [i for i in os.listdir(folder_path) if i!='.DS_Store']
    if descriptions is None:
        descriptions = {}
        for i in files_present:
            descriptions[i] = ''
    else:
        for i in files_present:
            if i not in descriptions.keys():
                descriptions[i] = ''
    #An entry in the descriptions dict is now present for all files. :)
    for i in files_present:
        file_path = folder_path+'/'+i
        EC_upload_file(private_api_key, dataset_name, file_path, \
                       description=descriptions[i], dev=dev)
    return

def EC_read_file(private_api_key, dataset_name, \
                     resource_name, datatype=None, dev=False):
    '''
    Data Type Options: None (returns as a string), 
                        json (returned json object),
                        csv (returned as a pd.DataFrame)
    '''
    if dev:
        datahub = 'dev-datahub.electrocat.org'
    else:
        datahub = 'datahub.electrocat.org'
    dataset_id, resource_id = EC_extract(private_api_key,dataset_name,resource_name)
    apiurl='https://'+datahub+'/api/3/action/'
    apiheader={ 'Authorization':str(private_api_key)}
    postheader={'authorization': str(private_api_key),\
                'content-type': 'application/json;charset=utf-8'}
    getfile = requests.get(apiurl+'resource_show?id='+resource_id,headers=apiheader,verify=False)
    if getfile.ok:
        filedata=json.loads(getfile.content)
        getfilepkg = requests.get(apiurl+'package_show?id='+filedata['result']['package_id'],\
                                headers=apiheader,verify=False)
        if getfilepkg.ok:
            pkgurl=filedata['result']['url']
            data = requests.get(pkgurl,\
                            params={'id':resource_id},headers=apiheader,verify=False).text
            if datatype == None:
                return data
            elif datatype == 'json':
                return json.loads(data)
            elif datatype == 'csv':
                return pd.read_csv(StringIO(data))
            else:
                raise ValueError('Invalid file type selected')
        else:
            raise ValueError('Package info is not accessible.')
    else:
        raise ValueError('Resource does not exist or is not accessible.')

def EC_read_folder(private_api_key, dataset_name, datatype=None, dev=False):
    '''
    Data Type Options: None (returns as a string), 
                        json (returned json object),
                        csv (returned as a pd.DataFrame)
    '''
    if dev:
        datahub = 'dev-datahub.electrocat.org'
    else:
        datahub = 'datahub.electrocat.org'
    dataset_ids, resource_ids = EC_extract(private_api_key,dataset_name)
    print('Number of files located in this dataset: '+str(len(resource_ids)))
    apiurl='https://'+datahub+'/api/3/action/'
    apiheader={ 'Authorization':str(private_api_key)}
    postheader={'authorization': str(private_api_key),\
                'content-type': 'application/json;charset=utf-8'}
    all_data_li = []
    for resource_id in resource_ids:
        getfile = requests.get(apiurl+'resource_show?id='+resource_id,headers=apiheader,verify=False)
        if getfile.ok:
            filedata=json.loads(getfile.content)
            getfilepkg = requests.get(apiurl+'package_show?id='+filedata['result']['package_id'],\
                                    headers=apiheader,verify=False)
            if getfilepkg.ok:
                pkgurl=filedata['result']['url']
                data = requests.get(pkgurl,\
                                params={'id':resource_id},headers=apiheader,verify=False).text
                if datatype == None:
                    all_data_li.append(data)
                elif datatype == 'json':
                    all_data_li.append(json.loads(data))
                elif datatype == 'csv':
                    all_data_li.append(pd.read_csv(StringIO(data)))
                elif datatype == 'SMART':
                    try:
                        all_data_li.append(json.loads(data))
                    except:
                        try:
                            df = pd.read_csv(StringIO(data))
                            if df.empty:
                                all_data_li.append(data)
                            else:
                                all_data_li.append(df)
                        except:
                            all_data_li.append(data)
                else:
                    raise ValueError('Invalid file type selected')
            else:
                print('Package corrupted: '+str(filedata))
        else:
            print('Resource does not exist or is not accessible: '+str(getfile))
    return all_data_li
