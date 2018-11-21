import numpy as np

def parameter_count(variable_list, full = False):
    parameter_table = []
    total_parameters = 0
    for variable in variable_list:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(variable.name)
        # print(shape)

        variable_parameters = 1
        for dim in shape:

            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
        parameter_table.append([variable.name, str(shape.as_list()), str(variable_parameters)])
    parameter_table.append(['total', '', str(total_parameters)])
    
    if (full):
        row_format = "{:<40s} {:<20s} {:<20s}"
        for row in parameter_table:
            print(row_format.format(*row))
    else:
        print('trainable parameters: ' + str(total_parameters))
        
def convert_to_onehot(labels, n_class):    
    onehot = np.eye(n_class)[np.squeeze(labels)]
    return onehot

def shuffle_together(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b