import os
import sys
import importlib


def main(config_path='', output_path='', solr_info=''):
    import nsidc_ftp_harvester
    nsidc_ftp_harvester = importlib.reload(nsidc_ftp_harvester)
    nsidc_ftp_harvester.nsidc_ftp_harvester(
        config_path=config_path, output_path=output_path, on_aws=False, solr_info=solr_info)


if __name__ == '__main__':
    main()
