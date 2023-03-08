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
    'cad_llm_v1': {
        # Added masks to padded input
        'modal': 'byt5-base',
        'batch_size': 4,
        'subset_range': [.2, .8],
        'max_length': 128,
    },
    'cad_llm_small_test': {
        'modal': 'byt5-small',
        'batch_size': 16,
        'subset_range': [0, 1],
        'max_length': 128,
    },
    'cad_llm_test': {
        'modal': 'google/byt5-base',
        'batch_size': 2,
        'subset_range': [0, 1],
        'max_length': 128,
    },
}
