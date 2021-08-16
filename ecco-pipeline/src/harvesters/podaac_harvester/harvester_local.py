def main(config, output_path, LOG_TIME, solr_info={}, grids_to_use=[]):
    import harvester
    return harvester.podaac_harvester(config, output_path, LOG_TIME, solr_info=solr_info,
                                      grids_to_use=grids_to_use)


if __name__ == '__main__':
    main()
