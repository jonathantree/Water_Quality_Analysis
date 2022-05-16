county_fips_dictionary={}
with open("State_County_FIPS.txt") as file:
    for line in file:
        (key,value) = line.strip().split("        ")
        county_fips_dictionary[int(key)] = value

    print("text file to county_fips_dictionary complete")
    
    first_value = list(county_fips_dictionary.values())[0]
    first_key = list(county_fips_dictionary.keys())[0]

    print('First Value: ', first_value)
    print('First Key: ', first_key)