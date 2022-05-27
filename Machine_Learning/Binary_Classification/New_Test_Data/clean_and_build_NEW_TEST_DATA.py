import pandas as pd
import os
import re


#Function that cleans the scraping data and builds/exports the final dataframe
def clean_scraped_data(state_dir):
    #Zip code info with county FIPS that we need to join to the final df
    zips_df = pd.read_csv(os.path.join(state_dir, 'zips.csv'), dtype={'zip':str, 'count_FIPS' : str})
    zips_df['county_fips'] = zips_df['county_fips'].apply('{:0>5}'.format)

    
    # Contaminant scraped data
    EWG_Scrape_df = pd.read_csv(os.path.join(state_dir, 'contaminants.csv'), on_bad_lines='skip')
    df = EWG_Scrape_df.copy()
    
    #Dataframe for utilities zip info
    utils_zip_df = pd.read_csv(os.path.join(state_dir, 'utilities.csv'), on_bad_lines='skip')
    df2 = utils_zip_df.copy()
    df2 = df2.rename(columns={'Utility name':'Utility'})
   
    #Assign a unique primary key to each utility
    df2['key'] = pd.factorize(df2['Utility'])[0]
    # Create a dictionay to map the primary key to the contaminants df    
    key_dict = df2.set_index('Utility').to_dict()['key']
    
    #Map the primary key to Contaminants df and reassign the datatype as int
    df['key'] = df['Utility'].map(key_dict)
    df['key'] = pd.Series(df['key'],dtype=pd.Int64Dtype())

    #Set the index of the dataframes to the primary key
    df = df.set_index('key')
    df2 = df2.set_index('key')
    
    #Do a left join on the primary key
    df3 = df.join(df2, lsuffix='_caller', rsuffix='_other')

    #Drop rows that have the NaN values that are present in rsuffix column
    df3 = df3.dropna(subset=['Utility_other'])
    
    #Drop rows with non EWG Guideline for a contaminant
    df3.dropna(subset=['EWG HEALTH GUIDELINE'], inplace=True)
    
    #Split of the contaminant measurements and units and people served
    df3['Units'] = df3['Utility Measuremnt'].apply(lambda x: x.split()[-1])
    df3['Utility Measuremnt'] = df3['Utility Measuremnt'].apply(lambda x: x.split()[0])
    df3['EWG HEALTH GUIDELINE'] = df3['EWG HEALTH GUIDELINE'].apply(lambda x: x.split()[0])
    df3['Legal Limit']=df3['Legal Limit'].apply(lambda x: '0 units' if pd.isnull(x) else x)
    df3['Legal Limit'] = df3['Legal Limit'].apply(lambda x: x.split()[0])
    df3['People served'] = df3['People served'].apply(lambda x: x.split(':')[-1])

    #Add a column for the state
    df3['State'] = state_dir
    
    #Replace any commas in the thousandths place
    df3.replace(',','', regex=True, inplace=True)
    
    #Change the datatype of the measurements and EWG Guidlines to numeric values
    df3['Utility Measuremnt'] = pd.to_numeric(df3['Utility Measuremnt'])
    df3['EWG HEALTH GUIDELINE'] = pd.to_numeric(df3['EWG HEALTH GUIDELINE'])
    df3['People served'] = pd.to_numeric(df3['People served'])

    #Define the Contaminant Factor (how many times larger is the utility measurement than EWG guideline)
    df3['Contaminant_Factor'] = df3['Utility Measuremnt']/df3['EWG HEALTH GUIDELINE']
    
    #Rename the column and drop the redundant 'Utility_other' column
    df3 = df3.rename(columns={'Utility_caller':'Utility'})
    df3 = df3.drop(columns=['Utility_other'])

    #Reorder Columns
    df3 = df3.reindex(columns=['State', 'City', 'Zip', 'Utility', 'People served', 'Contaminant', 'Utility Measuremnt', 'EWG HEALTH GUIDELINE',
       'Legal Limit', 'Units', 'Contaminant_Factor'])
    
    #Write cleaned dataset to state directory
    df3.to_csv(os.path.join(state_dir,'contaminants_cleaned.csv'), index=False)

    #Make a unique list of the Utilities to loop through next
    utlities_list = list(df3.Utility.unique())
    
    # Loop through the contaminants df, select a subset for each utility, calculate the sum of the contaminant factor
    # And generate a dictionary value for these data to be used in final dataset df
    new_dataset = []
    for utility in utlities_list:
        temp_df = df3[df3.Utility == utility]
        cont_factor_sum = round(temp_df.Contaminant_Factor.sum())
        zipcode = temp_df['Zip'].values[0]
        state = temp_df['State'].values[0]
        pop_served = temp_df['People served'].values[0]
        #print(f'The Contaminant factor for {utility} is: {cont_factor_sum}')
        new_dataset_dict = {
            'State': state,
            'Zip' : zipcode,
            'Utility' : utility,
            'Population Served': pop_served,
            'Contaminant_Factor' : cont_factor_sum
            
        }
        new_dataset.append(new_dataset_dict)
        
    new_dataset_df = pd.DataFrame(new_dataset)
    new_dataset_df.Zip = new_dataset_df.Zip.astype('str')
    
    # Create a dictionary to map the zip codes for each utility to the respective county FIPS code
    c_FIPS_dict = zips_df.set_index('zip').to_dict()['county_fips']
    new_dataset_df['county_FIPS'] = new_dataset_df['Zip'].map(c_FIPS_dict)
    
    #Save the resulting dataframe to a csv
    new_dataset_df.to_csv(os.path.join(state_dir,'cleaned.csv'), index=False)
      
    return new_dataset_df

#Iterate through every single state and run the cleaning funtion
clean_scraped_data( 'AR')

# print('Finished cleaning every state scraped data')
# print ('==============================================')
# print('Building the master dataset')

# #Build the master dataset from every state
# master_df = pd.DataFrame()
# for state_dir in os.listdir(data_dir):
#     d = os.path.join(data_dir, state_dir)
#     #print(d)
#     temp_df= pd.read_csv(os.path.join('AR', 'cleaned.csv' ))
#     #print(temp_df)
#     master_df = master_df.append(temp_df)

# #Pandas likes to try to reset the int FIPS code to float, change it back before writing the file
# master_df.county_FIPS = pd.Series(master_df.county_FIPS, dtype=pd.Int64Dtype())

# print('Writing the master utilitily dataset file')
# master_df.to_csv(os.path.join('AR', 'master_utilities.csv'), index=False)

#Write a master data file for all of the conaminants cleaned .csv files
# states_data = os.path.join('..','..','Resources','Data','user_scrape_data')
# list_of_available_data = []

# for state_dir in os.listdir(states_data):
#     list_of_available_data.append(state_dir)

# data_files = [os.path.join(states_data,i,'contaminants_cleaned.csv') for i in list_of_available_data]

# #combine all files in the list
# combined_df = pd.concat([pd.read_csv(f) for f in data_files ])

# print('Writing the master contaminants dataset file')
# combined_df.to_csv(os.path.join('..', '..' , 'Resources', 'Data', 'Cleaned_Data', 'all_contaminants.csv'), index=False)

# print('Done')

# pd.set_option('display.max_rows', 100)
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.width', 100)
# print('Here is the quality check on a sample of the master dataframe')
# print('_____________________________________________________________')
# print(new_dataset_df.sample(20))