filepath = '/Users/kevinmarlis/Developer/JPL/sealevel_output/gmsl_MERGED_TP_J1_OSTM_OST_GMSL_ASCII_V42/harvested_granules/1992/GMSL_TPJAOS_5.0_199209_202009.txt'

my_list = []

with open(filepath) as f:
    lines = f.readlines()

    data = {'altimeter type': [],
            'merged file cycle #': [],
            'year+fraction of year (mid-cycle)': [],
            'number of observations ': [],
            'number of weighted observations ': [],
            'GMSL (Global Isostatic Adjustment (GIA) not applied)': [],
            'standard deviation of GMSL (GIA not applied)': [],
            'smoothed (60-day Gaussian type filter) GMSL (GIA not applied)': [],
            'GMSL (Global Isostatic Adjustment (GIA) applied)': [],
            'standard deviation of GMSL (GIA applied)': [],
            'smoothed (60-day Gaussian type filter) GMSL (GIA applied)': [],
            'smoothed (60-day Gaussian type filter) GMSL (GIA applied) annual and semi-annual signal removed': [],
            }

    for line in lines:
        line = line.strip()
        if line:
            if line.startswith('HDR'):
                continue
            line = line.split()

            for (elem, column) in zip(line, data.keys()):
                data[column].append(elem)

ds = xr.Dataset(
    data_vars=dict(GMSL=(["time"], [float(i) for i in data['GMSL (Global Isostatic Adjustment (GIA) not applied)']]),
                   smoothed_GMSL=(["time"], [float(i) for i in data['smoothed (60-day Gaussian type filter) GMSL (GIA not applied)']])),
    coords=dict(time=[float(i)
                      for i in data['year+fraction of year (mid-cycle)']])
)

da = ds['GMSL']
da.plot()
