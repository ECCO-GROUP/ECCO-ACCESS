from ftplib import FTP
import time

user = 'anonymous'  # does not change
host = 'osisaf.met.no'  # does not change
dir = 'archive/ice/conc_amsr'
total = 0
ftp = FTP(host)
ftp.login(user)

years = [y.split('/')[-1] for y in ftp.nlst(dir)]
current = ftp.pwd()
for year in years:
    months = [m.split('/')[-1] for m in ftp.nlst(f'{dir}/{year}')]
    for month in months:
        print(ftp.nlst(f'{dir}/{year}/{month}'))

        exit()
        print(f'{year}-{month}-{len(ftp.nlst())}')
        total += int(len(ftp.nlst()))
        ftp.cwd(year_dir)
    ftp.cwd(current)
print(total)

overall_start = time.time()
host = 'sidads.colorado.edu'
dir = 'pub/DATASETS/NOAA/G02202_V3/south/daily/'
ftp = FTP(host)
ftp.login(user)
all_granules = []
years = ftp.nlst(dir)[2:]
for year in years:
    for hemi in ['north', 'south']:
        hemi_dir = dir.replace('south', hemi)
        gen = ftp.mlsd(f'{hemi_dir}{year}')
        for filename, details in gen:
            if details['type'] != 'file':
                continue
            granule = {
                'filename': filename,
                'filepath': f'{hemi_dir}{year}/{filename}',
                'size': details['size'],
                'modified': details['modify']
            }
            all_granules.append(granule)
overall_stop = time.time()
print(len(all_granules))
print('total time: ', overall_stop - overall_start)
