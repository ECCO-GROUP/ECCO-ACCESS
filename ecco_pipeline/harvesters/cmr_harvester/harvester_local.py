from harvesters.cmr_harvester import harvester


def main(config, output_path, grids_to_use=[]):
    return harvester.cmr_harvester(config, output_path, grids_to_use)


if __name__ == '__main__':
    main()
