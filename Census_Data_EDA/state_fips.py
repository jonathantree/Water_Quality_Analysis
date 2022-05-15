state_fips_dictionary={}
with open("Census_Data_EDA/State_FIPS.txt") as file:
    for line in file:
        (key,value) = line.strip().split("        ")
        state_fips_dictionary[int(key)] = value

    print("\ntext file to state_fips_dictionary=\n",state_fips_dictionary)