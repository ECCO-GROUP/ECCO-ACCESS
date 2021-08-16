def main(output_path, config, LOG_TIME, solr_info={}, grids_to_use=[]):
    import aggregation
    return aggregation.run_aggregation(
        output_path, config, LOG_TIME, solr_info=solr_info, grids_to_use=grids_to_use)


if __name__ == "__main__":
    main()
