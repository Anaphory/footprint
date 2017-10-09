#!/usr/bin/env python

import datetime

import json
import numpy
import pandas

import scipy.sparse
import scipy.optimize
import urllib
import pandasdmx

def atlas_api_call(model="hs07", export=True, year=datetime.date.today().year - 2, origin=None, destination=None, product=None):
    """Load export/import data from the OEC

    The Observatory of Economic Complexity atlas has a JSON API documented on
    http://atlas.media.mit.edu/api/. Query it for some data.

    """
    return json.load(
        urllib.request.urlopen(
            "http://atlas.media.mit.edu/{:}/{:}/{:}/{:}/{:}/{:}/".format(
                model, "export" if export else "import", year,
                "all" if origin is None else origin,
                "all" if destination is None else destination,
                "show" if product is None else product)))


# Some interesting data set URLs:
# worldbank_gdp_table: "http://databank.worldbank.org/data/download/GDP.csv"
# un statistics division, Worldbank world development indicator SDMX API: http://data.un.org/WS/rest/data/DF_UNDATA_WDI/A.AG_LND_AGRI_ZS.AFG+ALB/?lastNObservations=1'
# Many IO-Tables, some with high level of detail: https://www.bea.gov/industry/io_annual.htm and https://www.bea.gov/industry/io_benchmark.htm
# IO-Tables for the OECD countries. {:} can be one of the IDs below: "http://stats.oecd.org/SDMX-JSON/data/IOTS/{:}.NLD../all?detail=Full&dimensionAtObservation=AllDimensions&startPeriod=2008&endPeriod=2011"
# OECD inter-country IO tables: http://www.oecd.org/sti/ind/inter-country-input-output-tables.htm

[{'id': 'LEONTFD', 'name': 'Leontief inverse matrix (domestic)'},
 {'id': 'LEONTFT', 'name': 'Leontief inverse matrix (total)'},
 {'id': 'TTL', 'name': 'Total'},
 {'id': 'VAL', 'name': 'Value added'},
 {'id': 'DOMIMP', 'name': 'Domestic output and imports'},
 {'id': 'ICESHR', 'name': 'Imports content of exports, as % of exports'}]


# Obtain some country data relevant for ecological footprint calculations
unstats = pandasdmx.Request("UNSD")
for name, description in [
        ("AG_LND_AGRI_ZS", "Agricultural land (% of land area)"),
        ("AG_LND_FRST_K2", "Forest area (sq. km)"),
        ("EN_ATM_CO2E_PC", "CO2 emissions (metric tons per capita)"),
        ("NY_GDP_MKTP_CD", "GDP (current US$)"),
        ("NY_GNP_PCAP_CD", "GNI per capita, Atlas method (current US$)"),
        ("NY_GNP_PCAP_PP_CD", "GNI per capita, PPP (current international $)"),
        ("NY_GNP_ATLS_CD", "GNI, Atlas method (current US$)"),
        ("NY_GNP_MKTP_PP_CD", "GNI, PPP (current international $)"),
        ("AG_SRF_TOTL_K2", "Surface area (sq. km)")]:
    data = unstats.get(resource_type='data', resource_id="DF_UNDATA_WDI",
                       key={"FREQ": "A", "SERIES": name, "REF_AREA": []},
                       params={"lastNObservations": 1})

    sequence = data.write().max(axis=0)
    sequence.index = sequence.index.droplevel().droplevel()
    try:
        alldata[description] = sequence
    except NameError:
        alldata = pandas.DataFrame({description: sequence})

alldata["PPP"] = alldata["GNI per capita, Atlas method (current US$)"]/alldata["GNI per capita, PPP (current international $)"]
# Unit: US$/int$


# Read the ICIO (load it from server if unavailable) and generate a list of
# countries (with their PPPs and areas) and sectors
try:
    all_matrix = pandas.read_csv("./ICIO2016_2011.csv").set_index("Unnamed: 0")
except FileNotFoundError:
    from io import BytesIO
    import zipfile
    zipped = zipfile.ZipFile(BytesIO(
        urllib.request.urlopen(
            "http://www.oecd.org/sti/ind/ICIO2016_2011.zip").read()))
    zipped.extract("ICIO2016_2011.csv")
    all_matrix = pandas.read_csv("./ICIO2016_2011.csv").set_index("Unnamed: 0")
all_output = all_matrix.loc["OUT"]
sectors = [sect[4:] for sect in all_matrix.columns if sect.startswith("AUS_")]
countries = alldata.index
countryarea = alldata["Surface area (sq. km)"].values
# Unit: km²

# Now, I assume that the amount of land to create 1 int$ of goods/services is a
# meaningful number. I assume that it does depend on the sector, but not on the
# country. I assume that every (OECD) country essentially uses its entire land
# area, and none of its sea etc. area, in these industries, in weighted
# proportion to the sum of output they produce.
#
# That is, if O is the output matrix of the sectors by country, given in int$,
# and A is the vector of country land areas in km², then the minimum w of
# |O×w-A|² with 0≤w – with units km²/int$ – is a starting point for estimating
# the environmental footprint of goods and services known by industry sector,
# but not by production chain.
output_ppp = numpy.zeros((len(countries), len(sectors)))
for c, country in enumerate(countries):
    for s, sector in enumerate(sectors):
        try:
            output_ppp[c, s] = (all_output[country+"_"+sector] /
                                alldata["PPP"][country])
            # Unit: int$ = US$ / (US$/int$)
        except KeyError:
            print(country+"_"+sector)
            output_ppp[c, s] = numpy.nan

# Remove lines with missing data
output_ppp = output_ppp[~numpy.isnan(output_ppp).any(1)]

optimized = scipy.optimize.lsq_linear(output_ppp[:,:-1], output_ppp[:,-1], (0, numpy.inf))
# Unit: km²/int$

print(dict(zip(sectors, optimized.x)))
