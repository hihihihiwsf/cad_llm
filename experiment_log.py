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
        # Add masks to input and output, sort points in curves
        'modal': 'byt5-base',
        'batch_size': 12,
        'subset_range': [.2, .8],
        'max_length': 128,
        'eval_temperature': 1,
    },
    'cad_llm_no_pretrain_v1': {
        'modal': 'byt5-base-new',
        'batch_size': 12,
        'subset_range': [.2, .8],
        'max_length': 128,
        'eval_temperature': 1,
    },
    'cad_llm_q_tkz_v1': {
        'modal': 'byt5-base',
        'batch_size': 12,
        'subset_range': [.2, .8],
        'max_length': 128,
    },
    'cad_llm_q_tkz_lrn_v1': {
        'modal': 'byt5-base',
        'batch_size': 12,
        'subset_range': [.2, .8],
        'max_length': 128,
    },
}
