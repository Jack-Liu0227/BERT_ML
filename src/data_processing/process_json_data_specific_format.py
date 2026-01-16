#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
处理AlloyData.LT.json文件，提取HEA和MEA材料数据，将其转换为Data.training_998_HEAs.xlsx的格式
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from pathlib import Path
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 温度单位转换
def convert_temperature(value, unit):
    """
    将温度值从不同单位转换为开尔文(K)
    
    参数:
        value: 温度值
        unit: 温度单位，如'°C', 'K'等
        
    返回:
        转换后的开尔文温度值
    """
    if unit == "°C":
        return value + 273
    return value  # 默认返回原值，假设已经是开尔文

# 用于记录wt%到at%的转换
converted_compositions = []

# 定义默认单位
units = {
    'temperature': 'K',
    'time': 'hour',
    'size': 'mm',
    'composition': 'at.%',
    'yield_strength': 'MPa',
    'ultimate_strength': 'MPa',
    'elongation': '%',
    'impact_energy': 'J'
}

# 元素原子量字典，用于wt%到at%的转换
ATOMIC_WEIGHTS = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811, 'C': 12.011, 'N': 14.007, 
    'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 
    'Si': 28.086, 'P': 30.974, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 
    'Ca': 40.078, 'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938, 
    'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.390, 'Ga': 69.723, 
    'Ge': 72.640, 'As': 74.922, 'Se': 78.960, 'Br': 79.904, 'Kr': 83.800, 'Rb': 85.468, 
    'Sr': 87.620, 'Y': 88.906, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.940, 'Tc': 98.000, 
    'Ru': 101.070, 'Rh': 102.906, 'Pd': 106.420, 'Ag': 107.868, 'Cd': 112.411, 'In': 114.818, 
    'Sn': 118.710, 'Sb': 121.760, 'Te': 127.600, 'I': 126.905, 'Xe': 131.293, 'Cs': 132.906, 
    'Ba': 137.327, 'La': 138.906, 'Ce': 140.116, 'Pr': 140.908, 'Nd': 144.240, 'Pm': 145.000, 
    'Sm': 150.360, 'Eu': 151.964, 'Gd': 157.250, 'Tb': 158.925, 'Dy': 162.500, 'Ho': 164.930, 
    'Er': 167.259, 'Tm': 168.934, 'Yb': 173.040, 'Lu': 174.967, 'Hf': 178.490, 'Ta': 180.948, 
    'W': 183.840, 'Re': 186.207, 'Os': 190.230, 'Ir': 192.217, 'Pt': 195.078, 'Au': 196.967, 
    'Hg': 200.590, 'Tl': 204.383, 'Pb': 207.200, 'Bi': 208.980, 'Po': 209.000, 'At': 210.000, 
    'Rn': 222.000, 'Fr': 223.000, 'Ra': 226.000, 'Ac': 227.000, 'Th': 232.038, 'Pa': 231.036, 
    'U': 238.029, 'Np': 237.000, 'Pu': 244.000, 'Am': 243.000, 'Cm': 247.000, 'Bk': 247.000, 
    'Cf': 251.000, 'Es': 252.000, 'Fm': 257.000, 'Md': 258.000, 'No': 259.000, 'Lr': 262.000
}

# 定义目标格式的列名映射
HEA_COLUMNS_MAPPING = {
    # 基本属性
    'Number': 'Number',
    'YS(Mpa)': 'yield_strength',
    'UTS(Mpa)': 'ultimate_strength',
    'El(%)': 'elongation',
    
    # 化学元素含量
    'Al(at%)': 'Al',
    'Co(at%)': 'Co',
    'Cr(at%)': 'Cr',
    'Fe(at%)': 'Fe',
    'Ni(at%)': 'Ni',
    'Ti(at%)': 'Ti',
    'Ta(at%)': 'Ta',
    'V(at%)': 'V',
    'Mn(at%)': 'Mn',
    'Cu(at%)': 'Cu',
    'Mo(at%)': 'Mo',
    'C(at%)': 'C',
    'W(at%)': 'W',
    'Nb(at%)': 'Nb',
    
    # 处理条件
    'Hom_Temp(K)': 'homogenization_temperature',
    'CR(%)': 'cold_rolling_reduction',
    'recrystalize temperature/K': 'recrystallization_temperature',
    'recrystalize time/mins': 'recrystallization_time',
    'Anneal_Temp(K)': 'annealing_temperature',
    'Anneal_Time(h)': 'annealing_time',
    'aging temperature/K': 'aging_temperature',
    'aging time/hours': 'aging_time'
}

def extract_metadata(material):
    """
    提取元数据信息
    
    参数:
        material: 包含元数据的字典
        
    返回:
        包含元数据的字典
    """
    metadata = {}
    if 'metadata' in material:
        for meta_key, meta_value in material['metadata'].items():
            if not isinstance(meta_value, (list, dict)):
                metadata[f'{meta_key}'] = meta_value
    return metadata

def process_mech_prop(mech_prop, item_data):
    """
    处理机械性能数据
    
    参数:
        mech_prop: 机械性能数据字典
        item_data: 要更新的数据项字典
        
    返回:
        更新后的数据项字典
    """
    for key, value in mech_prop.items():
        item_data[f'{key}'] = value
        if key in units:
            item_data[f'{key}_unit'] = units[key]
    return item_data

def extract_element_content(composition, element):
    """
    从成分列表中提取特定元素的含量
    
    参数:
        composition: 成分列表，例如 [['Al', 3.6], ['Co', 27.3], ...]
        element: 要提取的元素符号，例如 'Al'
        
    返回:
        元素含量，如果元素不存在则返回0
    """
    for comp in composition:
        if isinstance(comp, list) and len(comp) >= 2 and comp[0] == element:
            return comp[1]
    return 0

def wt_to_at_percent(composition):
    """
    将重量百分比(wt%)转换为原子百分比(at%)
    
    参数:
        composition: 成分列表，例如 [['Al', 3.6], ['Co', 27.3], ...]
        
    返回:
        转换后的成分列表
    """
    # 计算原子分数
    atom_fractions = []
    total_atom_fraction = 0
    
    # 检查所有元素是否都在原子量表中
    valid_conversion = True
    missing_elements = []
    
    for element_data in composition:
        if isinstance(element_data, list) and len(element_data) >= 2:
            element = element_data[0]
            wt_percent = float(element_data[1])
            
            if element in ATOMIC_WEIGHTS:
                atom_weight = ATOMIC_WEIGHTS[element]
                atom_fraction = wt_percent / atom_weight
                atom_fractions.append([element, atom_fraction])
                total_atom_fraction += atom_fraction
            else:
                valid_conversion = False
                missing_elements.append(element)
    
    # 如果有元素缺失原子量，则记录警告并返回原始成分
    if not valid_conversion:
        logger.warning(f"无法转换wt%到at%: 缺少元素原子量 {missing_elements}")
        return composition
    
    # 计算原子百分比
    at_composition = []
    for element_atom_fraction in atom_fractions:
        element = element_atom_fraction[0]
        atom_fraction = element_atom_fraction[1]
        at_percent = (atom_fraction / total_atom_fraction) * 100
        at_composition.append([element, round(at_percent, 2)])  # 保留两位小数
    
    return at_composition

def process_materials(materials, item_data):
    """
    处理材料信息
    
    参数:
        materials: 材料信息字典
        item_data: 要更新的数据项字典
        
    返回:
        更新后的数据项字典
    """
    global converted_compositions
    
    for key, value in materials.items():
        if key != 'composition':
            item_data[f'{key}'] = value
    
    # 处理比例类型
    composition_unit = materials.get('ratio_type', units.get('composition', 'at.%'))
    item_data['ratio_type'] = composition_unit
    
    # 处理化学成分 
    if 'composition' in materials:
        comp = materials['composition']
        logger.debug(f"Composition data: {comp}")
        
        # 如果是wt%，则转换为at%
        if composition_unit == 'wt.%':
            # logger.info(f"转换成分从wt%到at%: {comp}")
            original_comp = str(comp)  # 保存原始成分以供记录
            comp = wt_to_at_percent(comp)
            # logger.info(f"转换后的成分(at%): {comp}")
            
            # 记录转换结果
            converted_compositions.append({
                '原始比例类型': composition_unit,
                '原始成分': original_comp,
                '转换后成分': str(comp)
            })
            
            # 更新为at%
            item_data['ratio_type'] = 'at.%'
        
        # 构建完整的成分字符串
        composition_str = []
        for element in comp:
            if isinstance(element, list) and len(element) >= 2:
                composition_str.append(f"{element[0]}{element[1]}")
        
        item_data['composition'] = "".join(composition_str)
        item_data['composition_at'] = "".join(composition_str)
        
        # 提取各元素含量
        elements = ['Al', 'Co', 'Cr', 'Fe', 'Ni', 'Ti', 'Ta', 'V', 'Mn', 'Cu', 'Mo', 'C', 'W', 'Nb']
        for element in elements:
            item_data[element] = extract_element_content(comp, element)
    
    return item_data

def process_processing(proc, item_data):
    """
    处理加工信息，将加工步骤转换为一个描述性文本，并提取特定处理参数
    
    参数:
        proc: 加工信息字典
        item_data: 要更新的数据项字典
        
    返回:
        更新后的数据项字典
    """
    # 初始化处理参数
    process_params = {
        'homogenization_temperature': None,
        'cold_rolling_reduction': None,
        'recrystallization_temperature': None,
        'recrystallization_time': None,
        'annealing_temperature': None,
        'annealing_time': None,
        'aging_temperature': None,
        'aging_time': None
    }
    
    # 处理非proc_para和surf_para的键
    for key, value in proc.items():
        if key not in ['proc_para', 'surf_para']:
            item_data[f'processing_{key}'] = value
    
    # 处理加工步骤
    if 'proc_para' in proc:
        # 记录加工类型
        proc_types = []
        for step in proc['proc_para']:
            if 'type' in step:
                proc_type = step['type'].lower()
                proc_types.append(step['type'])
                
                # 提取均质化处理温度
                if 'homog' in proc_type or 'solution' in proc_type:
                    if 'temperature' in step and isinstance(step['temperature'], dict) and 'value' in step['temperature']:
                        temp_value = step['temperature']['value']
                        temp_unit = step['temperature'].get('unit', 'K')
                        
                        if isinstance(temp_value, list) and len(temp_value) > 0:
                            temp = convert_temperature(temp_value[0], temp_unit)
                            process_params['homogenization_temperature'] = temp
                        else:
                            temp = convert_temperature(temp_value, temp_unit)
                            process_params['homogenization_temperature'] = temp
                
                # 提取冷轧减薄率
                if 'roll' in proc_type and 'cold' in proc_type:
                    if 'reduction' in step:
                        if isinstance(step['reduction'], dict) and 'value' in step['reduction']:
                            red_value = step['reduction']['value']
                            if isinstance(red_value, list) and len(red_value) > 0:
                                process_params['cold_rolling_reduction'] = red_value[0]
                            else:
                                process_params['cold_rolling_reduction'] = red_value
                        else:
                            process_params['cold_rolling_reduction'] = step.get('reduction')
                
                # 提取再结晶温度和时间
                if 'recryst' in proc_type:
                    if 'temperature' in step and isinstance(step['temperature'], dict) and 'value' in step['temperature']:
                        temp_value = step['temperature']['value']
                        temp_unit = step['temperature'].get('unit', 'K')
                        
                        if isinstance(temp_value, list) and len(temp_value) > 0:
                            temp = convert_temperature(temp_value[0], temp_unit)
                            process_params['recrystallization_temperature'] = temp
                        else:
                            temp = convert_temperature(temp_value, temp_unit)
                            process_params['recrystallization_temperature'] = temp
                    
                    if 'time' in step and isinstance(step['time'], dict) and 'value' in step['time']:
                        time_value = step['time']['value']
                        if isinstance(time_value, list) and len(time_value) > 0:
                            # 转换为分钟
                            minutes = time_value[0]
                            if 'unit' in step['time'] and step['time']['unit'] == 'hour':
                                minutes = time_value[0] * 60
                            process_params['recrystallization_time'] = minutes
                        else:
                            # 转换为分钟
                            minutes = time_value
                            if 'unit' in step['time'] and step['time']['unit'] == 'hour':
                                minutes = time_value * 60
                            process_params['recrystallization_time'] = minutes
                
                # 提取退火温度和时间
                if 'anneal' in proc_type:
                    if 'temperature' in step and isinstance(step['temperature'], dict) and 'value' in step['temperature']:
                        temp_value = step['temperature']['value']
                        temp_unit = step['temperature'].get('unit', 'K')
                        
                        if isinstance(temp_value, list) and len(temp_value) > 0:
                            temp = convert_temperature(temp_value[0], temp_unit)
                            process_params['annealing_temperature'] = temp
                        else:
                            temp = convert_temperature(temp_value, temp_unit)
                            process_params['annealing_temperature'] = temp
                    
                    if 'time' in step and isinstance(step['time'], dict) and 'value' in step['time']:
                        time_value = step['time']['value']
                        if isinstance(time_value, list) and len(time_value) > 0:
                            process_params['annealing_time'] = time_value[0]
                        else:
                            process_params['annealing_time'] = time_value
                
                # 提取时效处理温度和时间
                if 'ag' in proc_type:
                    if 'temperature' in step and isinstance(step['temperature'], dict) and 'value' in step['temperature']:
                        temp_value = step['temperature']['value']
                        temp_unit = step['temperature'].get('unit', 'K')
                        
                        if isinstance(temp_value, list) and len(temp_value) > 0:
                            temp = convert_temperature(temp_value[0], temp_unit)
                            process_params['aging_temperature'] = temp
                        else:
                            temp = convert_temperature(temp_value, temp_unit)
                            process_params['aging_temperature'] = temp
                    
                    if 'time' in step and isinstance(step['time'], dict) and 'value' in step['time']:
                        time_value = step['time']['value']
                        if isinstance(time_value, list) and len(time_value) > 0:
                            process_params['aging_time'] = time_value[0]
                        else:
                            process_params['aging_time'] = time_value
        
        item_data['processing_types'] = "; ".join(proc_types)
        
        # 将提取的处理参数添加到item_data
        for param_key, param_value in process_params.items():
            if param_value is not None:
                item_data[param_key] = param_value
        
        # 使用generate_processing_description生成描述
        try:
            processing_description = generate_processing_description(proc)
            item_data['processing_description'] = processing_description
        except Exception as e:
            logger.warning(f"生成加工描述时出错: {str(e)}")
            item_data['processing_description'] = f"处理过程包括: {', '.join(proc_types)}"
    
    return item_data

def process_testing(testing, item_data):
    """
    处理测试信息
    
    参数:
        testing: 测试信息字典
        item_data: 要更新的数据项字典
        
    返回:
        更新后的数据项字典
    """
    for key, value in testing.items():
        if key != 'rate':
            if isinstance(value, list):
                # 将列表转换为字符串，用逗号分隔
                value_str = ", ".join(map(str, [v for v in value if v is not None]))
                item_data[f'testing_{key}'] = value_str
                
                # 特殊处理测试温度 - 去除方括号
                if key == 'test_tem':
                    # 直接使用数值或提取单值
                    if len(value) == 1:
                        item_data['temperature'] = value[0]  # 直接存储数值而非字符串
                    else:
                        item_data['temperature'] = value_str  # 多个值时保持逗号分隔字符串
                    
                    # 处理温度单位转换
                    if 'test_tem_unit' in testing and testing['test_tem_unit'] == '°C':
                        if len(value) == 1:
                            item_data['temperature'] = convert_temperature(value[0], '°C')
                        else:
                            # 多个温度值时，需要分别转换
                            temps = [convert_temperature(float(t.strip()), '°C') for t in value_str.split(',')]
                            item_data['temperature'] = ", ".join(map(str, temps))
                    
                    item_data['testing_test_tem_unit'] = units.get('temperature', 'K')
            else:
                item_data[f'testing_{key}'] = value
                
                # 特殊处理单个测试温度值
                if key == 'test_tem':
                    item_data['temperature'] = value  # 直接存储值
                    
                    # 处理温度单位转换
                    if 'test_tem_unit' in testing and testing['test_tem_unit'] == '°C':
                        item_data['temperature'] = convert_temperature(value, '°C')
                    
                    item_data['testing_test_tem_unit'] = units.get('temperature', 'K')
    
    # 处理rate
    if 'rate' in testing and testing['rate']:
        for i, rate in enumerate(testing['rate']):
            if isinstance(rate, dict):
                if 'value' in rate:
                    val_str = ", ".join(map(str, rate['value'])) if isinstance(rate['value'], list) else str(rate['value'])
                    item_data[f'testing_rate{i+1}_value'] = val_str
                if 'unit' in rate:
                    item_data[f'testing_rate{i+1}_unit'] = rate['unit']
    
    return item_data

def generate_processing_description(proc):
    """
    生成加工过程的描述性文本
    
    参数:
        proc: 加工信息字典
        
    返回:
        加工过程描述文本
    """
    description = []
    
    if 'proc_para' in proc:
        for i, step in enumerate(proc['proc_para']):
            step_desc = []
            
            # 添加步骤类型
            if 'type' in step:
                step_desc.append(f"{step['type']}")
            
            # 添加温度信息
            if 'temperature' in step:
                temp = step['temperature']
                if isinstance(temp, dict) and 'value' in temp:
                    temp_values = temp['value'] if isinstance(temp['value'], list) else [temp['value']]
                    temp_unit = temp.get('unit', units.get('temperature', 'K'))
                    
                    # 如果温度单位是摄氏度，则转换为开尔文
                    if temp_unit == '°C':
                        temp_values = [convert_temperature(t, '°C') for t in temp_values]
                        temp_unit = 'K'  # 转换后统一使用K单位
                    
                    temp_str = ", ".join(map(str, temp_values))
                    step_desc.append(f"at {temp_str} {temp_unit}")
            
            # 添加时间信息
            if 'time' in step:
                time = step['time']
                if isinstance(time, dict) and 'value' in time:
                    time_values = time['value'] if isinstance(time['value'], list) else [time['value']]
                    time_unit = time.get('unit', units.get('time', 'hour'))
                    time_str = ", ".join(map(str, time_values))
                    step_desc.append(f"for {time_str} {time_unit}")
            
            # 添加其他信息
            for key, value in step.items():
                if key not in ['type', 'temperature', 'time'] and not isinstance(value, dict):
                    step_desc.append(f"{key}: {value}")
            
            description.append(" ".join(step_desc))
    
    return "; ".join(description)

def json_to_dataframe(data, mat_types=['HEA', 'MEA']):
    """
    将JSON数据转换为DataFrame，仅提取指定材料类型的数据
    
    参数:
        data: JSON数据
        mat_types: 要提取的材料类型列表，默认为HEA和MEA
        
    返回:
        转换后的DataFrame
    """
    all_data = []
    
    # 处理L-T部分
    if 'L-T' in data and "articles" in data['L-T']:
        for material in data['L-T']["articles"]:
            # 提取元数据
            metadata = extract_metadata(material)
            
            # 检查是否存在scidata和datasets
            if 'scidata' in material and 'datasets' in material['scidata']:
                for dataset in material['scidata']['datasets']:
                    # 先检查是否为目标材料类型
                    is_target_material = False
                    if 'materials' in dataset and 'mat_type' in dataset['materials']:
                        mat_type = dataset['materials']['mat_type']
                        if mat_type in mat_types:
                            is_target_material = True
                            # 记录材料类型到item_data
                            metadata['mat_type'] = mat_type
                    
                    # 如果不是目标材料，则跳过
                    if not is_target_material:
                        continue
                    
                    # 创建当前数据集的字典
                    item_data = metadata.copy()
                    
                    # 基本dataset信息
                    if 'list' in dataset:
                        item_data['list'] = dataset['list']
                    
                    # 处理各部分数据
                    if 'mech_prop' in dataset:
                        item_data = process_mech_prop(dataset['mech_prop'], item_data)
                    
                    if 'materials' in dataset:
                        item_data = process_materials(dataset['materials'], item_data)
                    
                    if 'processing' in dataset:
                        item_data = process_processing(dataset['processing'], item_data)
                    
                    if 'testing' in dataset:
                        item_data = process_testing(dataset['testing'], item_data)
                    
                    # 处理score
                    if 'score' in dataset:
                        item_data['score'] = dataset['score']
                    
                    # 添加到总列表
                    all_data.append(item_data)
    else:
        logger.warning("JSON数据中无法找到'L-T'部分或'articles'列表")
    
    # 创建DataFrame并处理NaN值
    df = pd.DataFrame(all_data)
    df = df.replace({np.nan: None})
    
    return df

def map_to_hea_format(df):
    """
    将处理后的DataFrame转换为Data.training_998_HEAs_1.csv的格式
    
    参数:
        df: 输入DataFrame
        
    返回:
        转换后的DataFrame
    """
    # 创建一个新的DataFrame来存储转换后的数据
    hea_data = []
    
    # 添加列头，确保与Data.training_998_HEAs_1.csv完全一致
    header = {
        '': '',
        'temperature': '',
        'Property': 'Property',
        'Unnamed: 2': '',
        'Unnamed: 3': '',
        'Chemical compositions': 'Chemical compositions',
        'Unnamed: 5': '',
        'Unnamed: 6': '',
        'Unnamed: 7': '',
        'Unnamed: 8': '',
        'Unnamed: 9': '',
        'Unnamed: 10': '',
        'Unnamed: 11': '',
        'Unnamed: 12': '',
        'Unnamed: 13': '',
        'Unnamed: 14': '',
        'Unnamed: 15': '',
        'Unnamed: 16': '',
        'Unnamed: 17': '',
        'Unnamed: 18': '',
        'Processing actions': 'Processing actions',
        'Unnamed: 20': '',
        'Unnamed: 21': '',
        'Unnamed: 22': '',
        'Unnamed: 23': '',
        'Unnamed: 24': '',
        'Unnamed: 25': '',
        'Unnamed: 26': ''
    }
    hea_data.append(header)
    
    # 添加第二行表头，与Data.training_998_HEAs_1.csv保持一致
    subheader = {
        '': 'Number',
        'temperature': 'Temperature(K)',
        'Property': 'YS(Mpa)',
        'Unnamed: 2': 'UTS(Mpa)',
        'Unnamed: 3': 'El(%)',
        'Chemical compositions': 'Al(at%)',
        'Unnamed: 5': 'Co(at%)',
        'Unnamed: 6': 'Cr(at%)',
        'Unnamed: 7': 'Fe(at%)',
        'Unnamed: 8': 'Ni(at%)',
        'Unnamed: 9': 'Ti(at%)',
        'Unnamed: 10': 'Ta(at%)',
        'Unnamed: 11': 'V(at%)',
        'Unnamed: 12': 'Mn(at%)',
        'Unnamed: 13': 'Cu(at%)',
        'Unnamed: 14': 'Mo(at%)',
        'Unnamed: 15': 'C(at%)',
        'Unnamed: 16': 'V(at%)',
        'Unnamed: 17': 'W(at%)',
        'Unnamed: 18': 'Nb(at%)',
        'Processing actions': 'Hom_Temp(K)',
        'Unnamed: 20': 'CR(%)',
        'Unnamed: 21': 'recrystalize temperature/K',
        'Unnamed: 22': 'recrystalize time/mins',
        'Unnamed: 23': 'Anneal_Temp(K)',
        'Unnamed: 24': 'Anneal_Time(h)',
        'Unnamed: 25': 'aging temperature/K',
        'Unnamed: 26': 'aging time/hours'
    }
    hea_data.append(subheader)
    
    # 添加数据行
    for i, row in df.iterrows():
        # 确保aging_time正确转换为小时
        aging_time = 0
        if 'aging_time' in row and row['aging_time'] is not None:
            # 检查单位，如果需要转换
            if 'aging_time_unit' in row and row['aging_time_unit'] == 'min':
                aging_time = float(row['aging_time']) / 60  # 分钟转小时
            else:
                aging_time = float(row['aging_time'])  # 假设已经是小时
        
        # 设置默认温度值为室温(298K)
        default_temp = 298
        
        row_data = {
            '': i + 1,  # 为每行分配一个序号
            'temperature': row.get('temperature', default_temp),
            'Property': row.get('yield_strength', ''),
            'Unnamed: 2': row.get('ultimate_strength', ''), 
            'Unnamed: 3': row.get('elongation', ''),
            'Chemical compositions': row.get('Al', 0),
            'Unnamed: 5': row.get('Co', 0),
            'Unnamed: 6': row.get('Cr', 0),
            'Unnamed: 7': row.get('Fe', 0),
            'Unnamed: 8': row.get('Ni', 0),
            'Unnamed: 9': row.get('Ti', 0),
            'Unnamed: 10': row.get('Ta', 0),
            'Unnamed: 11': row.get('V', 0),
            'Unnamed: 12': row.get('Mn', 0),
            'Unnamed: 13': row.get('Cu', 0),
            'Unnamed: 14': row.get('Mo', 0),
            'Unnamed: 15': row.get('C', 0),
            'Unnamed: 16': row.get('V', 0),  # 重复V数据（按照样本格式）
            'Unnamed: 17': row.get('W', 0),
            'Unnamed: 18': row.get('Nb', 0),
            'Processing actions': row.get('homogenization_temperature', default_temp),
            'Unnamed: 20': row.get('cold_rolling_reduction', 0),
            'Unnamed: 21': row.get('recrystallization_temperature', default_temp),
            'Unnamed: 22': row.get('recrystallization_time', 0),
            'Unnamed: 23': row.get('annealing_temperature', default_temp),
            'Unnamed: 24': row.get('annealing_time', 0),
            'Unnamed: 25': row.get('aging_temperature', default_temp),
            'Unnamed: 26': aging_time,
        }
        hea_data.append(row_data)
    
    # 创建新的DataFrame
    result_df = pd.DataFrame(hea_data)
    
    # 确保列的顺序与Data.training_998_HEAs_1.csv一致，添加temperature列
    column_order = ['', 'temperature', 'Property', 'Unnamed: 2', 'Unnamed: 3', 
                    'Chemical compositions', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 
                    'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 
                    'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 
                    'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18', 
                    'Processing actions', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 
                    'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26']
    
    # 确保DataFrame中包含所有需要的列
    for col in column_order:
        if col not in result_df.columns:
            result_df[col] = ''
    
    # 按照指定顺序重新排列列
    result_df = result_df[column_order]
    
    # 自定义保存CSV的函数，确保每个字段后面都有逗号（与目标格式一致）
    def custom_to_csv(df, path, **kwargs):
        with open(path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                line = ','.join([str(v) if v is not None else '' for v in row.values]) + ','
                f.write(line + '\n')
    
    # 覆盖标准to_csv方法
    result_df.to_csv = lambda path, **kwargs: custom_to_csv(result_df, path, **kwargs)
    
    return result_df

def process_alloy_json(input_json_path, output_excel_path=None, mat_types=['HEA', 'MEA']):
    """
    处理合金JSON数据文件，提取指定材料类型，并转换为指定格式
    
    参数:
        input_json_path: 输入JSON文件路径
        output_excel_path: 输出Excel文件路径，默认为None（自动基于输入文件名生成）
        mat_types: 要处理的材料类型列表，默认为HEA和MEA
        
    返回:
        生成的Excel文件路径
    """
    logger.info(f"开始处理JSON文件: {input_json_path}, 目标材料类型: {mat_types}")
    
    # 确定输出路径
    if output_excel_path is None:
        output_excel_path = str(Path(input_json_path).with_suffix('.processed.xlsx'))
    
    # 加载JSON数据
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"成功加载JSON数据，大小: {len(str(data))} 字符")
    except Exception as e:
        logger.error(f"加载JSON数据时出错: {str(e)}")
        return None
    
    # 转换为DataFrame，仅包含指定材料类型
    df = json_to_dataframe(data, mat_types)
    logger.info(f"成功转换为DataFrame，包含 {len(df)} 行和 {len(df.columns)} 列")
    
    # 如果未找到指定材料类型数据，则返回
    if len(df) == 0:
        logger.warning(f"未找到指定材料类型数据: {mat_types}")
        return None
    
    # 转换为目标格式
    hea_df = map_to_hea_format(df)
    logger.info(f"成功转换为目标格式，包含 {len(hea_df)} 行")
    
    # 保存为Excel
    try:
        # Save HEA format data to Excel
        hea_df.to_excel(output_excel_path, index=False, engine='openpyxl')
        logger.info(f"已成功将HEA格式数据保存为Excel: '{output_excel_path}'")
        csv_path = str(Path(output_excel_path).with_suffix('.csv'))
        hea_df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"已成功将HEA格式数据保存为CSV: '{csv_path}'")
        # Save original data to CSV with same base name
        csv_path = str(Path(output_excel_path).with_name(Path(output_excel_path).stem + '_source.csv'))
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"已保存原始数据为CSV格式: '{csv_path}'")
        
        return output_excel_path
    except Exception as e:
        logger.error(f"保存Excel文件时出错: {str(e)}")
        return None

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='处理合金JSON数据，提取指定类型的材料数据并转换为目标格式')
    parser.add_argument('--input', '-i', type=str, help='输入JSON文件路径',
                        default=str(Path(__file__).parent.parent.parent / "datasets" / "AlloyData.LT.json"))
    parser.add_argument('--output', '-o', type=str, help='输出Excel文件路径',
                        default=str(Path(__file__).parent.parent.parent / "datasets" / "HEA_MEA_data.xlsx"))
    parser.add_argument('--mat-types', '-m', type=str, nargs='+', default=['HEA', 'MEA'],
                        help='要处理的材料类型，默认为HEA和MEA')
    parser.add_argument('--conversion-log', '-c', type=str, help='wt%到at%的转换日志文件路径',
                        default=str(Path(__file__).parent.parent.parent / "datasets" / "conversion_log.csv"))
    args = parser.parse_args()
    
    # 记录要处理的材料类型
    logger.info(f"要处理的材料类型: {args.mat_types}")
    
    # 处理JSON数据
    result_path = process_alloy_json(args.input, args.output, args.mat_types)
    
    if result_path:
        logger.info(f"处理完成！结果保存在: {result_path}")
        
        # 如果有转换日志，保存到CSV
        if converted_compositions:
            conversion_log_df = pd.DataFrame(converted_compositions, 
                                             columns=['原始比例类型', '原始成分', '转换后成分'])
            conversion_log_df.to_csv(args.conversion_log, index=False, encoding='utf-8')
            logger.info(f"wt%到at%的转换日志保存在: {args.conversion_log}")
    else:
        logger.error("处理失败！")

if __name__ == "__main__":
    main() 