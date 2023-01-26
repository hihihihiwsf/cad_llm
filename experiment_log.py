"""
Keep track of experiments we run
"""

experiment_name_to_hps = {
    'cad_llm_v0': {
        'modal': 'byt5-base',
        'batch_size': 2,
        'subset_range': [0, 1],
    },
    'cad_llm_no_pretrain_v0': {
        'modal': 'byt5-base-new',
        'batch_size': 2,
        'subset_range': [0, 1],
    },
    # 'cad_llm_v1': {
    #     'modal': 'byt5-base',
    #     'batch_size': 4,
    #     'subset_range': [.25, .75],
    #     'max_length': 128,
    # },
}
