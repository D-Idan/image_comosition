from pathlib import Path


path_current_file = Path(__file__).parent.absolute()

path_val = path_current_file / 'val'
path_val = path_val if path_val.exists() else (_ for _ in ()).throw(ValueError('Validation CIFAR Path does not exist'))
# path_test = Path('./test') if Path('./test').exists() else Raise(ValueError('Test CIFAR Path does not exist'))

path_train_original = path_current_file / 'original/test'
path_train_original = path_train_original if path_train_original.exists() else (_ for _ in ()).throw(ValueError('Original CIFAR Path does not exist'))

path_train_one_percent = path_current_file / 'one_percent/test'
path_train_one_percent = path_train_one_percent if path_train_one_percent.exists() else (_ for _ in ()).throw(ValueError('one_percent CIFAR Path does not exist'))

path_train_double_one_percent = path_current_file / 'double_one_percent/test'
path_train_double_one_percent = path_train_double_one_percent if path_train_double_one_percent.exists() else (_ for _ in ()).throw(ValueError('double_one_percent CIFAR Path does not exist'))
