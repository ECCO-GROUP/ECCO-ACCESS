from harvesters.osisaf_ftp_harvester import harvester


def main(config, output_path, grids_to_use=[]):
    return harvester.osisaf_ftp_harvester(config, output_path, grids_to_use)


if __name__ == '__main__':
    main()
