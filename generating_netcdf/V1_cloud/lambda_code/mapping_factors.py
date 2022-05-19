import pickle

def get_mapping_factors(ea, mapping_factors_dir, debug_mode, source_grid_all, target_grid, target_grid_radius, source_grid_min_L, source_grid_max_L, source_grid_k, nk):
    print('\nGrid Mappings')
    grid_mapping_fname = mapping_factors_dir / "ecco_latlon_grid_mappings.p"

    if debug_mode:
        print('...DEBUG MODE -- SKIPPING GRID MAPPINGS')
        grid_mappings_all = []
        grid_mappings_k = []
    else:

        if 'grid_mappings_k' not in globals():

            # first check to see if you have already calculated the grid mapping factors
            if grid_mapping_fname.is_file():
                # if so, load
                print('... loading latlon_grid_mappings.p')

                [grid_mappings_all, grid_mappings_k] = pickle.load(open(grid_mapping_fname, 'rb'))

            else:
                # if not, make new grid mapping factors
                print('... no mapping factors found, recalculating')

                # find the mapping between all points of the ECCO grid and the target grid.
                grid_mappings_all = \
                    ea.find_mappings_from_source_to_target(source_grid_all,
                                                            target_grid,
                                                            target_grid_radius,
                                                            source_grid_min_L,
                                                            source_grid_max_L)

                # then find the mapping factors between all wet points of the ECCO grid
                # at each vertical level and the target grid
                grid_mappings_k = dict()

                for k in range(nk):
                    print(k)
                    grid_mappings_k[k] = \
                        ea.find_mappings_from_source_to_target(source_grid_k[k],
                                                                target_grid,
                                                                target_grid_radius,
                                                                source_grid_min_L,
                                                                source_grid_max_L)
                if not mapping_factors_dir.exists():
                    try:
                        mapping_factors_dir.mkdir()
                    except:
                        print ('cannot make %s ' % mapping_factors_dir)

                try:
                    pickle.dump([grid_mappings_all, grid_mappings_k], open(grid_mapping_fname, 'wb'))
                except:
                    print('cannot make %s ' % mapping_factors_dir)
        else:
            print('... grid mappings k already in memory')  

    return (grid_mappings_all, grid_mappings_k)