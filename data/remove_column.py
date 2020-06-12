import csv
with open("Covid-19-R.csv","r") as source:
    rdr= csv.reader( source )
    with open("Covid-19-R-cleaned.csv","w") as result:
        wtr= csv.writer( result )
        for r in rdr:
        	#print(r[5])
            wtr.writerow( (r[1], r[2], r[3], r[4], r[6]) )