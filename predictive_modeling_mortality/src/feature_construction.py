import utils
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn import preprocessing


class feature_construction(object):
    
    def __init__(self):       
        train_path = '../data/train/'
        events, mortality, feature_map = self.read_csv(train_path)
        patient_features, mortality = self.create_features(events, mortality, feature_map)
        self.save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')
        
        
    def read_csv(self,filepath):          
        events = pd.read_csv(filepath + 'events.csv')
        mortality = pd.read_csv(filepath + 'mortality_events.csv')
        feature_map = pd.read_csv(filepath + 'event_feature_map.csv')
        
        return events, mortality, feature_map
 

    def calculate_index_date(self, events, mortality):  
        #create two groups based on patients that are alive and those that are deceased
        alive_dates = events[['patient_id','timestamp']].loc[~events['patient_id'].isin(mortality['patient_id'])].groupby(['patient_id']).max().reset_index()
        dead_dates = mortality[['patient_id','timestamp']]

        # index date for deceased
        dead_dates['timestamp']= dead_dates['timestamp'].apply(pd.to_datetime) - timedelta(days = 30)

        # combine two groups with the patient ids and index dates
        indx_date = pd.concat([alive_dates,dead_dates]).reset_index(drop=True)
        indx_date.columns = ['patient_id','indx_date']

        return indx_date


    def filter_events(self, events, indx_date):
        # get all events, convert timestamp to useable format
        events_merge = pd.merge(events,indx_date, on = ['patient_id'])
        events_merge[['timestamp','indx_date']] = events_merge[['timestamp','indx_date']].apply(pd.to_datetime)

        #filter out events that do not meet the observation period criteria
        filtered_events = events_merge[(events_merge['timestamp'] <= events_merge['indx_date']) & (events_merge['timestamp'] >= (events_merge['indx_date']-timedelta(days = 2000)))]
        filtered_events = filtered_events[['patient_id','event_id','value']]

        return filtered_events

    
    def aggregate_events(self,filtered_events_df, mortality_df,feature_map_df):
        #create column to indicate aggregation group
        filtered_events_df['event_type'] = filtered_events_df['event_id'].apply(lambda x: x[0])
        #replace events with event index
        idx_events = pd.merge(filtered_events_df,feature_map_df, on = 'event_id')
        #remove events with na values
        idx_events = idx_events[idx_events['value'].notna()]

        #DIAG/DRUG sum group - sum of diagnostic/medication event values
        d_events = idx_events[idx_events['event_type']=='D']
        d_events = d_events.groupby(['patient_id','idx'])[['value']].sum()
        d_events.reset_index(inplace = True)

        #LAB count group - count of lab event occurences
        l_events = idx_events[idx_events['event_type']=='L']
        l_events = l_events.groupby(['patient_id','idx'])[['value']].count()
        l_events.reset_index(inplace = True)

        #prep for export 
        aggregated_events = pd.concat([d_events,l_events]).reset_index(drop = True)   
        aggregated_events.columns = ['patient_id','feature_id','feature_value']
        aggregated_events['feature_value'] = aggregated_events['feature_value'].round(6)

        #normalize different events to the same scale using min-max normalization
        pivot = aggregated_events.pivot(index='patient_id', columns='feature_id', values='feature_value')
        norm = pivot/pivot.max()
        aggregated_events = pd.melt(norm.reset_index(), id_vars='patient_id',
                                    value_name='feature_value').dropna()

        return aggregated_events  
    
    
    def create_features(self, events, mortality, feature_map):
        #Calculate index date
        indx_date = self.calculate_index_date(events, mortality)

        #Filter events in the observation window
        filtered_events = self.filter_events(events, indx_date)

        #Aggregate the event values for each patient 
        aggregated_events = self.aggregate_events(filtered_events, mortality, feature_map)

        '''
        Create two dictionaries:
        1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
        2. mortality : Key - patient_id and value is mortality label
        '''       
        patient_features = aggregated_events
        patient_features['zipped'] = list(zip(patient_features.feature_id, patient_features.feature_value ))
        patient_features = patient_features.drop(['feature_id','feature_value'], axis = 1)
        patient_features  = {k : list(v) for k,v in patient_features.groupby('patient_id')['zipped']}

        mortality = mortality.drop('timestamp',axis = 1)
        mortality = pd.Series(mortality.label.values, index = mortality.patient_id).to_dict()
        
        return patient_features, mortality   
               
    def save_svmlight(self, patient_features, mortality, op_file, op_deliverable):
        deliverable1 = open(op_file, 'w')
        deliverable2 = open(op_deliverable, 'w')

        for key in sorted(patient_features):
            if key in mortality:
                line1 = "%d" %(1)
                line2 = "%d %d" %(key,1)
            else:
                line1 = "%d" %(0)
                line2 = "%d %d" %(key,0)
            for value in sorted(patient_features[key]):
                pairs = "%d:%.7f" %(value[0], value[1])
                line1 = line1 + " "+pairs+" "
                line2 = line2 + " "+pairs+" "
            deliverable1.write(line1+"\n")
            deliverable2.write(line2+"\n")