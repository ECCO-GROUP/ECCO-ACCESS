from harvesters.podaac_harvester import harvester


def main(config, output_path, grids_to_use=[]):
    return harvester.podaac_harvester(config, output_path, grids_to_use)


if __name__ == '__main__':
    main()
