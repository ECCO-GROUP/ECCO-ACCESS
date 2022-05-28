import os
import lzma
import pickle


def create_mapping_factors(ea, dataset_dim, mapping_factors_dir, debug_mode, source_grid_all, target_grid, target_grid_radius, source_grid_min_L, source_grid_max_L, source_grid_k, nk):
    print('\nCreating Grid Mappings')

    if not mapping_factors_dir.exists():
        try:
            mapping_factors_dir.mkdir()
        except:
            print ('cannot make %s ' % mapping_factors_dir)
    
    grid_mapping_fname_all = mapping_factors_dir / 'ecco_latlon_grid_mappings_all.xz'
    grid_mapping_fname_2D = mapping_factors_dir / 'ecco_latlon_grid_mappings_2D.xz'
    grid_mapping_fname_3D = mapping_factors_dir / '3D'

    if not grid_mapping_fname_3D.exists():
        try:
            grid_mapping_fname_3D.mkdir()
        except:
            print ('cannot make %s ' % grid_mapping_fname_3D)

    if debug_mode:
        print('...DEBUG MODE -- SKIPPING GRID MAPPINGS')
        grid_mappings_all = []
        grid_mappings_k = []
    else:
        # first check to see if you have already calculated the grid mapping factors
        if dataset_dim == '3D':
            all_3D = True
            all_3D_fnames = [f'ecco_latlon_grid_mappings_3D_{i}.xz' for i in range(nk)]
            curr_3D_fnames = os.listdir(grid_mapping_fname_3D)
            for fname in all_3D_fnames:
                if fname not in curr_3D_fnames:  
                    all_3D = False
                    break

        if (dataset_dim == '2D' and grid_mapping_fname_2D.is_file()) or (dataset_dim == '3D' and all_3D):
            # Factors already made, continuing
            print('... mapping factors already created')

        else:
            # if not, make new grid mapping factors
            print('... no mapping factors found, recalculating')

            if ~(grid_mapping_fname_all.is_file()):
                # find the mapping between all points of the ECCO grid and the target grid.
                grid_mappings_all = \
                    ea.find_mappings_from_source_to_target(source_grid_all,
                                                            target_grid,
                                                            target_grid_radius,
                                                            source_grid_min_L,
                                                            source_grid_max_L)

                # Save grid_mappings_all
                try:
                    pickle.dump(grid_mappings_all, lzma.open(grid_mapping_fname_all, 'wb'))
                except:
                    print('cannot make %s ' % grid_mapping_fname_all)

            # If the dataset is 2D, only compute one level of the mapping factors
            if dataset_dim == '2D':
                nk = 1

            # Find the mapping factors between all wet points of the ECCO grid
            # at each vertical level and the target grid
            for k_i in range(nk):
                print(k_i)
                grid_mappings_k = \
                    ea.find_mappings_from_source_to_target(source_grid_k[k_i],
                                                            target_grid,
                                                            target_grid_radius,
                                                            source_grid_min_L,
                                                            source_grid_max_L)

                try:
                    if dataset_dim == '2D':
                        pickle.dump(grid_mappings_k, lzma.open(grid_mapping_fname_2D, 'wb'))
                    elif dataset_dim == '3D':
                        fname_3D = grid_mapping_fname_3D / f'ecco_latlon_grid_mappings_3D_{k_i}.xz'
                        pickle.dump(grid_mappings_k, lzma.open(fname_3D, 'wb'))
                except:
                    print('cannot make %s ' % mapping_factors_dir)
    return


def get_mapping_factors(dataset_dim, mapping_factors_dir, factors_to_get, debug_mode=False, extra_prints=False, k=0):
    # factors_to_get : factors to load in from the mapping_factors_dir
    # can be 'all', 'k', or 'both'
    grid_mappings_all = []
    grid_mappings_k = []

    if extra_prints: print('\nGetting Grid Mappings')
    grid_mapping_fname_all = mapping_factors_dir / 'ecco_latlon_grid_mappings_all.xz'
    grid_mapping_fname_2D = mapping_factors_dir / 'ecco_latlon_grid_mappings_2D.xz'
    grid_mapping_fname_3D = mapping_factors_dir / '3D' / f'ecco_latlon_grid_mappings_3D_{k}.xz'

    if debug_mode:
        print('...DEBUG MODE -- SKIPPING GRID MAPPINGS')
        grid_mappings_all = []
        grid_mappings_k = []
    else:
        # Check to see that the mapping factors have been made
        if (dataset_dim == '2D' and grid_mapping_fname_2D.is_file()) or (dataset_dim == '3D' and grid_mapping_fname_3D.is_file()):
            # if so, load
            try:
                if factors_to_get == 'all' or factors_to_get == 'both':
                    if extra_prints: print(f'... loading ecco_latlon_grid_mappings_all.xz')
                    grid_mappings_all = pickle.load(lzma.open(grid_mapping_fname_all, 'rb'))

                if factors_to_get == 'k' or factors_to_get == 'both':
                    if dataset_dim == '2D':
                        if extra_prints: print(f'... loading ecco_latlon_grid_mappings_{dataset_dim}.xz')
                        grid_mappings_k = pickle.load(lzma.open(grid_mapping_fname_2D, 'rb'))
                    elif dataset_dim == '3D':
                        if extra_prints: print(f'... loading ecco_latlon_grid_mappings_{dataset_dim}_{k}.xz')
                        grid_mappings_k = pickle.load(lzma.open(grid_mapping_fname_3D, 'rb'))
            except:
                print(f'Unable to load grid mapping factors: {mapping_factors_dir}')
        else:
            print(f'Grid mapping factors have not been created or cannot be found: {mapping_factors_dir}')

    return (grid_mappings_all, grid_mappings_k)