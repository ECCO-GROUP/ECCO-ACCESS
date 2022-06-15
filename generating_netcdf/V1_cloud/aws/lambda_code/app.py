import time
import importlib

def run_script(event, context):
    script = importlib.import_module('eccov4r4_gen_for_podaac_cloud')

    output_freq_code = event['output_freq_code']
    product_type = event['product_type']
    grouping_to_process = event['grouping_to_process']
    time_steps_to_process = event['time_steps_to_process']
    config_metadata = event['config_metadata']
    aws_metadata = event['aws_metadata']
    debug_mode = event['debug_mode']
    local = event['local']
    credentials = event['credentials']

    script.generate_netcdfs(
        output_freq_code,
        product_type,
        grouping_to_process,
        time_steps_to_process,
        config_metadata,
        aws_metadata,
        debug_mode,
        local,
        credentials
    )

    return


def handler(event, context):
    print('Inside handler')

    run_script(event, context)
    
    return