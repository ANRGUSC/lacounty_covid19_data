#/bin/sh
echo -n "Are the scripts updated with number of days (y/n)? "
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
	echo "copying files"
else
	echo "Exiting"
	exit 0
fi
cp Covid-19.csv ../../../covid19_risk_estimation/data/
cp Covid-19.csv ../../../covid19_risk_estimation/software/raw_python_scripts/
cp Covid-19-R.csv ../../../covid19_risk_estimation/software/raw_python_scripts/
cp Covid-19-R.csv ../../../covid19_risk_estimation/data/
cp Covid-19-density.csv ../../../covid19_risk_estimation/data/
cp Covid-19-density.csv  ../../../covid19_risk_estimation/software/raw_python_scripts/
cp lacounty_covid.json ../../../covid19_risk_estimation/data/
cp lacounty_covid.json ../../../covid19_risk_estimation/software/raw_python_scripts/


