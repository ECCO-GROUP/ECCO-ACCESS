def main(config, output_path, solr_info={}, grids_to_use=[]):
    import harvester
    harvester.osisaf_ftp_harvester(config, output_path, solr_info=solr_info,
                                   grids_to_use=grids_to_use)


if __name__ == '__main__':
    main()
