#!/bin/bash

#/home5/ifenty/no_backup/podaac_20201216/lat-lon/day_mean

#python create_thumbnails.py  --dataset_base_dir='/home5/ifenty/no_backup/podaac_20201216/native/day_mean/' --product_type='native' --output_dir='/home5/ifenty/no_backup/podaac_20201216/thumbnails/native/day_mean/' --grouping_to_process='all' --thumbnail_height=9.11
#python create_thumbnails.py  --dataset_base_dir='/home5/ifenty/no_backup/podaac_20201216/native/mon_mean/' --product_type='native' --output_dir='/home5/ifenty/no_backup/podaac_20201216/thumbnails/native/mon_mean/' --grouping_to_process='all' --thumbnail_height=9.11
#python create_thumbnails.py  --dataset_base_dir='/home5/ifenty/no_backup/podaac_20201216/native/day_inst/' --product_type='native' --output_dir='/home5/ifenty/no_backup/podaac_20201216/thumbnails/native/day_inst/' --grouping_to_process='all' --thumbnail_height=9.11
#python create_thumbnails.py  --dataset_base_dir='/home5/ifenty/no_backup/podaac_20201216/lat-lon/day_mean/' --product_type='latlon' --output_dir='/home5/ifenty/no_backup/podaac_20201216/thumbnails/lat-lon/day_mean/' --grouping_to_process='all' --thumbnail_height=18.0

python create_thumbnails.py  --dataset_base_dir='/home5/ifenty/no_backup/podaac_20201216/lat-lon/mon_mean/' --product_type='latlon' --output_dir='/home5/ifenty/no_backup/podaac_20201216/thumbnails/lat-lon/mon_mean/' --grouping_to_process='all' --thumbnail_height=18.0
