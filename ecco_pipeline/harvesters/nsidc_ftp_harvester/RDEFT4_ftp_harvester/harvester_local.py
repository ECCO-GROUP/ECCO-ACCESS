from harvesters.nsidc_ftp_harvester.RDEFT4_ftp_harvester import harvester


def main(config, output_path, grids_to_use=[]):
    return harvester.seaice_harvester(config, output_path, grids_to_use)


if __name__ == '__main__':
    main()
