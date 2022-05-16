state_fips_dictionary={}
with open("State_FIPS.txt") as file:
    for line in file:
        (key,value) = line.strip().split("        ")
        state_fips_dictionary[int(key)] = value

    print(f"\ntext file to state_fips_dictionary=\n",state_fips_dictionary)