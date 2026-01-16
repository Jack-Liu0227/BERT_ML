#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Constants for alloy data processing
合金数据处理常量
"""

# Common element symbols
COMMON_ELEMENTS = [
    'Al', 'Co', 'Cr', 'Cu', 'Fe', 'Mn', 'Mo', 'Ni', 'Ti', 'V', 'W', 'Zr'
]

# Default temperature ranges (°C)
DEFAULT_TEMP_RANGES = {
    'room_temp': (20, 30),
    'low_temp': (-196, 0),
    'high_temp': (500, 1000),
    'very_high_temp': (1000, 2000)
}

# Default composition ranges (%)
DEFAULT_COMPOSITION_RANGES = {
    'low': (0, 5),
    'medium': (5, 20),
    'high': (20, 50),
    'very_high': (50, 100)
}

# Default property ranges
DEFAULT_PROPERTY_RANGES = {
    'tensile_strength': (200, 2000),  # MPa
    'yield_strength': (100, 1500),    # MPa
    'elongation': (0, 50),            # %
    'hardness': (100, 800),           # HV
    'fracture_toughness': (10, 100),  # MPa·m^1/2
    'corrosion_rate': (0, 10),        # mm/year
    'wear_rate': (0, 1)               # mm^3/N·m
}

# Common processing methods
PROCESSING_METHODS = [
    'casting',
    'powder_metallurgy',
    'mechanical_alloying',
    'arc_melting',
    'induction_melting',
    'laser_cladding',
    'additive_manufacturing'
]

# Common heat treatment processes
HEAT_TREATMENTS = [
    'annealing',
    'quenching',
    'tempering',
    'aging',
    'solution_treatment',
    'precipitation_hardening'
]

# Common atmosphere types
ATMOSPHERES = [
    'air',
    'argon',
    'nitrogen',
    'vacuum',
    'hydrogen',
    'helium'
]

# Common property types
PROPERTY_TYPES = [
    'tensile_strength',
    'yield_strength',
    'elongation',
    'hardness',
    'fracture_toughness',
    'corrosion_resistance',
    'wear_resistance'
]

# File extensions
FILE_EXTENSIONS = {
    'csv': '.csv',
    'json': '.json',
    'excel': '.xlsx'
}

# Default column names
DEFAULT_COLUMNS = {
    'composition': 'composition',
    'temperature': 'temperature',
    'time': 'time',
    'pressure': 'pressure',
    'atmosphere': 'atmosphere',
    'method': 'method',
    'property': 'property',
    'value': 'value',
    'unit': 'unit'
}

# Default units
DEFAULT_UNITS = {
    'temperature': '°C',
    'time': 'h',
    'pressure': 'MPa',
    'composition': '%',
    'strength': 'MPa',
    'elongation': '%',
    'hardness': 'HV'
} 