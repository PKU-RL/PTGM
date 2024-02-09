from gym3.types import TensorType, Real

POLICY_KWARGS = {
    "action_space": TensorType(shape=(512,), eltype=Real()),
    "policy_kwargs": {
        'attention_heads': 16, 
        'attention_mask_style': 'clipped_causal', 
        'attention_memory_size': 256, 
        'diff_mlp_embedding': False, 
        'hidsize': 2048, 
        'img_shape': [128, 128, 3], 
        'impala_chans': [16, 32, 32], 
        'impala_kwargs': {'post_pool_groups': 1}, 
        'impala_width': 8, 
        'init_norm_kwargs': {'batch_norm': False, 'group_norm_groups': 1}, 
        'n_recurrence_layers': 4, 
        'only_img_input': True, 
        'pointwise_ratio': 4, 
        'pointwise_use_activation': False, 
        'recurrence_is_residual': True, 
        'recurrence_type': 'transformer', 
        'timesteps': 128, 
        'use_pointwise_layer': True, 
        'use_pre_lstm_ln': False},
    "pi_head_kwargs": {'temperature': 1.0}
}
