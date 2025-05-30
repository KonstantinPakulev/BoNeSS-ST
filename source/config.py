import os

import source.namespace as m_ns
import source.method.ness.criterion.namespace as c_ns


METHOD_NAME2ALIAS = {
    m_ns.SHITOMASI: 'Shi-Tomasi',
    m_ns.SIFT: 'SIFT',

    m_ns.SUPERPOINT: 'SuperPoint',
    m_ns.R2D2: 'R2D2',
    m_ns.KEYNET: 'Key.Net',
    m_ns.DISK: 'DISK',
    m_ns.REKD: 'REKD',
    m_ns.NESSST: 'NeSS-ST',
    m_ns.HARDNET: 'HardNet',

    m_ns.SSST: 'SS-ST'
}

DEFAULT = 'default'


class MethodConfig:

    @staticmethod
    def from_rel_path(rel_path, ablation_name):
        split_list = rel_path.split('/')
        method_name_idx = [i for i in range(0, len(split_list)) if split_list[i] in m_ns.METHOD_LIST]

        method_prefix = '/'.join(split_list[1:method_name_idx[0]])

        method_base_name_list = [split_list[i] for i in method_name_idx]
        method_base_name = "_".join(method_base_name_list)

        rel_path_suffix_split_list = [i for i in split_list[method_name_idx[-1] + 1:]]

        method_name_suffix_split_list = [i.replace('_', '') for i in rel_path_suffix_split_list]

        if DEFAULT in method_name_suffix_split_list:
            method_name_suffix_split_list.remove(DEFAULT)

        if ablation_name is not None:
            ablation_idx = method_name_suffix_split_list.index(ablation_name.replace('_', ''))

            version_suffix = ''.join(method_name_suffix_split_list[:ablation_idx])
            ablation_suffix = ''.join(method_name_suffix_split_list[ablation_idx:])

        else:
            if len(method_name_suffix_split_list) == 0:
                version_suffix = None
                ablation_suffix = None

            else:
                version_suffix = ''.join(method_name_suffix_split_list)
                ablation_suffix = None

        suffix_rel_path = '/'.join(rel_path_suffix_split_list)

        return MethodConfig(method_base_name, version_suffix, ablation_suffix, suffix_rel_path, method_prefix)

    @staticmethod
    def from_evaluation_rel_path(method_evaluation_rel_path, ablation_name):
        split_list = method_evaluation_rel_path.split('/')
        split_list[0] = split_list[0].replace('_', '/')

        if len(split_list) == 1:
            return MethodConfig.from_rel_path(os.path.join(split_list[0], DEFAULT), ablation_name)

        else:
            return MethodConfig.from_rel_path(os.path.join(split_list[0], '/'.join(split_list[1:])), ablation_name)

    def __init__(self, method_base_name, version_suffix, ablation_suffix,
                 suffix_rel_path, method_prefix=None):
        self.method_base_name = method_base_name
        self.version_suffix = version_suffix
        self.ablation_suffix = ablation_suffix

        self.method_name = compose_method_name(self.method_base_name, self.version_suffix, self.ablation_suffix)

        self.suffix_rel_path = suffix_rel_path

        self.method_prefix = method_prefix

    def get_method_name(self):
        return self.method_name

    def get_suffix_rel_path(self):
        return self.suffix_rel_path

    def get_method_base_name(self):
        return self.method_base_name

    def get_method_config_rel_path(self):
        return os.path.join('config/model',
                            self.method_prefix if self.method_prefix is not None else '',
                            self.get_method_base_name().replace('_', '/'),
                            self.get_suffix_rel_path())

    def get_parent_method_config(self):
        if self.is_ablation():
            split_list = self.suffix_rel_path.split('/')

            return MethodConfig(self.method_base_name, self.version_suffix, None,
                                '/'.join(split_list[:-2]))

        else:
            if self.is_baseline():
                return None

            else:
                suffix_rel_path_split_list = self.suffix_rel_path.split('/')

                suffix_rel_path_split_list.pop(-1)

                if len(suffix_rel_path_split_list) != 0:
                    version_suffix = ''.join([i.replace('_', '') for i in suffix_rel_path_split_list])
                    suffix_rel_path = '/'.join(suffix_rel_path_split_list)

                    return MethodConfig(self.method_base_name, version_suffix, None,
                                        suffix_rel_path)

                else:
                    return MethodConfig(self.method_base_name, None, None,
                                        DEFAULT)

    def get_alias(self):
        method_base_name_split_list = self.method_base_name.split('_')
        version_suffix = '' if self.version_suffix is None else f'#{self.version_suffix}'
        ablation_suffix = '' if self.ablation_suffix is None else f'@{self.ablation_suffix}'

        if len(method_base_name_split_list) == 1:
            return METHOD_NAME2ALIAS[method_base_name_split_list[0]]

        else:
            return f'{METHOD_NAME2ALIAS[method_base_name_split_list[0]]}{version_suffix}{ablation_suffix}+' \
                   f'{METHOD_NAME2ALIAS[method_base_name_split_list[1]]}'

    def is_baseline(self):
        return self.method_base_name.split('_')[0] in m_ns.BASELINE_METHOD_LIST and \
               self.version_suffix is None

    def is_ablation(self):
        return self.ablation_suffix is not None

    def reinitialize(self, descriptor_name):
        method_base_name_split_list = self.method_base_name.split('_')

        method_base_name = f'{method_base_name_split_list[0]}_{descriptor_name}'

        return MethodConfig(method_base_name,
                            self.version_suffix,
                            self.ablation_suffix,
                            self.suffix_rel_path,
                            self.method_prefix)


"""
Support utils
"""


def remove_ablation_from_suffix_rel_path(suffix_rel_path, ablation_suffix):
    char_remains = len(ablation_suffix)

    split_list = suffix_rel_path.split('/')
    method_name_split_list = suffix_rel_path.replace('_', '').split('/')

    for i in range(len(method_name_split_list) - 1, -1, -1):
        char_remains -= len(method_name_split_list[i])

        if char_remains == 0:
            break

        elif char_remains < 0:
            raise ValueError('Incorrect tracing:', suffix_rel_path, ablation_suffix)

    return '/'.join(split_list[:i])


def compose_method_name(method_base_name, version_suffix, ablation_suffix):
    method_base_name_list = method_base_name.split('_')
    version_suffix = '' if version_suffix is None else f'#{version_suffix}'
    ablation_suffix = '' if ablation_suffix is None else f'@{ablation_suffix}'

    method_name = f"{method_base_name_list[0]}{version_suffix}{ablation_suffix}"

    if len(method_base_name_list) != 1:
        method_name = f"{method_name}_{method_base_name_list[1]}"

    return method_name


def decompose_model_name(model_name):
    model_name_splits = model_name.split('_')

    if len(model_name_splits) == 1:
        return model_name_splits[0], None, None

    else:
        model_name1 = model_name_splits[0]
        model_name_splits1 = model_name1.split('#')

        if len(model_name_splits1) == 1:
            return model_name, None, None

        else:
            model_suffix1 = model_name_splits1[1]
            model_suffix_splits1 = model_suffix1.split('@')

            base_model_name = '_'.join([model_name_splits1[0], model_name_splits[1]])

            if len(model_suffix_splits1) == 3:
                return base_model_name, '@'.join(model_suffix_splits1[:2]), model_suffix_splits1[2]

            else:
                return base_model_name, model_suffix1, None
