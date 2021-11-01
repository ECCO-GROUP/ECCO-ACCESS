from aggregation_by_year import aggregation


def main(output_path, config, grids_to_use=[]):
    return aggregation.run_aggregation(output_path, config, grids_to_use)


if __name__ == "__main__":
    main()
