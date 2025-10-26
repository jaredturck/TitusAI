d_model = 1024
nhead = d_model // 64
dim_feedforward = d_model * 4
no_transformer_layers = d_model // 128
dropout = 0.1
embedding_size = 259
max_length = 200
batch_size = 260

def transformer_param_count():

    total = (embedding_size * d_model) + (max_length * d_model) + (no_transformer_layers *
        (
            (
                (3 * d_model * d_model) + \
                (3 * d_model) + \
                (d_model * d_model) + d_model
            ) + (
                (d_model * dim_feedforward + dim_feedforward) + \
                (dim_feedforward * d_model + d_model)
            ) + (4 * d_model)
        )
    ) + embedding_size

    print(f'{total:,} parameters, approximate training tokens {(total * 20) / 1_000_000:,.2f} million')

transformer_param_count()
