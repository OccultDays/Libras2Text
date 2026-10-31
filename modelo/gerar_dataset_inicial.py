import csv
import math
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "servidor"))
from extrator_pontos import calcular_vetor_quadro, montar_vetor_temporal

MOLDES_MAOS = {
    "A_pdf": [
        [-0.17, -0.127], [-0.245, -0.361], [-0.223, -0.558], [-0.185, -0.691], [-0.118, -0.506], [-0.066, -0.682], [-0.107, -0.579], [-0.141, -0.476], [-0.0, -0.466], [0.046, -0.641],
        [-0.027, -0.505], [-0.067, -0.395], [0.113, -0.404], [0.139, -0.546], [0.061, -0.397], [0.023, -0.288], [0.228, -0.323], [0.234, -0.436], [0.159, -0.334], [0.115, -0.242],
    ],
    "A_ines": [
        [-0.181, -0.165], [-0.241, -0.38], [-0.232, -0.565], [-0.272, -0.694], [-0.111, -0.502], [-0.085, -0.643], [-0.112, -0.531], [-0.123, -0.422], [-0.0, -0.464], [0.032, -0.588],
        [-0.017, -0.466], [-0.045, -0.363], [0.104, -0.409], [0.125, -0.529], [0.067, -0.418], [0.033, -0.319], [0.208, -0.342], [0.208, -0.446], [0.152, -0.373], [0.117, -0.293],
    ],
    "B_pdf": [
        [-0.136, -0.12], [-0.217, -0.289], [-0.151, -0.431], [-0.036, -0.458], [-0.097, -0.474], [-0.071, -0.643], [-0.045, -0.745], [-0.018, -0.834], [-0.0, -0.472], [0.029, -0.658],
        [0.054, -0.783], [0.081, -0.886], [0.079, -0.435], [0.115, -0.612], [0.133, -0.726], [0.153, -0.82], [0.158, -0.37], [0.188, -0.507], [0.192, -0.597], [0.196, -0.676],
    ],
    "B_ines": [
        [-0.124, -0.12], [-0.159, -0.284], [-0.125, -0.404], [-0.084, -0.465], [-0.096, -0.465], [-0.07, -0.672], [-0.049, -0.793], [-0.026, -0.89], [0.0, -0.466], [0.029, -0.686],
        [0.055, -0.827], [0.082, -0.934], [0.089, -0.435], [0.117, -0.642], [0.142, -0.77], [0.169, -0.868], [0.174, -0.378], [0.204, -0.529], [0.222, -0.63], [0.24, -0.716],
    ],
    "C_pdf": [
        [0.169, -0.009], [0.346, -0.074], [0.504, -0.128], [0.634, -0.192], [0.193, -0.413], [0.364, -0.609], [0.514, -0.62], [0.613, -0.588], [0.168, -0.428], [0.356, -0.614],
        [0.504, -0.618], [0.602, -0.567], [0.175, -0.413], [0.343, -0.565], [0.469, -0.565], [0.558, -0.535], [0.187, -0.394], [0.329, -0.498], [0.43, -0.505], [0.504, -0.486],
    ],
    "C_ines": [
        [-0.218, -0.027], [-0.386, -0.105], [-0.517, -0.136], [-0.633, -0.141], [-0.318, -0.371], [-0.371, -0.534], [-0.447, -0.575], [-0.521, -0.584], [-0.263, -0.385], [-0.31, -0.552],
        [-0.395, -0.589], [-0.482, -0.59], [-0.211, -0.385], [-0.258, -0.539], [-0.336, -0.593], [-0.411, -0.607], [-0.161, -0.373], [-0.198, -0.506], [-0.259, -0.567], [-0.322, -0.597],
    ],
    "D_pdf": [
        [0.137, -0.055], [0.271, -0.182], [0.41, -0.29], [0.473, -0.413], [-0.031, -0.44], [-0.079, -0.671], [-0.122, -0.831], [-0.158, -0.951], [-0.0, -0.466], [0.167, -0.662],
        [0.336, -0.605], [0.411, -0.499], [0.046, -0.458], [0.213, -0.63], [0.344, -0.56], [0.387, -0.463], [0.076, -0.427], [0.194, -0.562], [0.287, -0.547], [0.328, -0.49],
    ],
    "D_ines": [
        [-0.139, -0.171], [-0.198, -0.334], [-0.239, -0.446], [-0.234, -0.544], [-0.059, -0.503], [0.079, -0.663], [0.14, -0.734], [0.203, -0.782], [-0.0, -0.48], [0.034, -0.652],
        [-0.085, -0.634], [-0.155, -0.565], [0.05, -0.432], [0.057, -0.587], [-0.069, -0.569], [-0.148, -0.508], [0.092, -0.367], [0.106, -0.507], [0.005, -0.526], [-0.075, -0.503],
    ],
    "E_pdf": [
        [-0.178, -0.101], [-0.284, -0.269], [-0.272, -0.41], [-0.152, -0.4], [-0.107, -0.482], [-0.057, -0.67], [-0.09, -0.62], [-0.11, -0.522], [0.0, -0.464], [0.072, -0.679],
        [0.026, -0.62], [-0.007, -0.513], [0.09, -0.422], [0.161, -0.61], [0.109, -0.575], [0.069, -0.478], [0.181, -0.368], [0.232, -0.511], [0.191, -0.514], [0.146, -0.461],
    ],
    "E_ines": [
        [-0.199, -0.105], [-0.31, -0.253], [-0.334, -0.372], [-0.248, -0.42], [-0.114, -0.49], [-0.06, -0.675], [-0.071, -0.663], [-0.082, -0.584], [0.0, -0.473], [0.075, -0.668],
        [0.057, -0.637], [0.03, -0.548], [0.097, -0.421], [0.171, -0.595], [0.151, -0.578], [0.123, -0.494], [0.187, -0.345], [0.26, -0.486], [0.252, -0.498], [0.228, -0.454],
    ],
    "F_pdf": [
        [0.14, -0.185], [0.184, -0.359], [0.228, -0.48], [0.287, -0.562], [0.024, -0.528], [0.264, -0.586], [0.411, -0.549], [0.494, -0.511], [0.0, -0.492], [0.274, -0.579],
        [0.402, -0.544], [0.482, -0.505], [0.024, -0.423], [0.223, -0.61], [0.344, -0.742], [0.436, -0.845], [0.085, -0.346], [0.139, -0.455], [0.161, -0.546], [0.205, -0.631],
    ],
    "F_ines": [
        [-0.149, -0.19], [-0.19, -0.386], [-0.2, -0.526], [-0.238, -0.621], [-0.087, -0.461], [-0.202, -0.54], [-0.314, -0.553], [-0.415, -0.55], [-0.0, -0.479], [0.02, -0.69],
        [0.012, -0.852], [0.018, -0.976], [0.08, -0.447], [0.179, -0.659], [0.227, -0.799], [0.269, -0.906], [0.144, -0.372], [0.291, -0.5], [0.381, -0.591], [0.45, -0.668],
    ],
    "I_pdf": [
        [-0.145, -0.125], [-0.199, -0.316], [-0.106, -0.461], [0.023, -0.485], [-0.115, -0.468], [-0.065, -0.607], [-0.083, -0.483], [-0.108, -0.403], [0.0, -0.449], [0.058, -0.565],
        [0.01, -0.417], [-0.032, -0.346], [0.113, -0.418], [0.168, -0.516], [0.09, -0.39], [0.032, -0.309], [0.215, -0.38], [0.277, -0.531], [0.302, -0.628], [0.331, -0.734],
    ],
    "I_ines": [
        [-0.125, -0.187], [-0.153, -0.369], [-0.167, -0.508], [-0.162, -0.624], [-0.066, -0.484], [-0.105, -0.617], [-0.147, -0.516], [-0.162, -0.42], [-0.0, -0.453], [-0.047, -0.572],
        [-0.102, -0.459], [-0.114, -0.353], [0.076, -0.427], [0.018, -0.533], [-0.048, -0.428], [-0.071, -0.327], [0.154, -0.395], [0.183, -0.563], [0.19, -0.676], [0.202, -0.771],
    ],
    "L_pdf": [
        [-0.171, -0.09], [-0.322, -0.219], [-0.469, -0.294], [-0.594, -0.319], [-0.124, -0.491], [-0.127, -0.714], [-0.131, -0.848], [-0.122, -0.955], [0.0, -0.474], [0.02, -0.616],
        [-0.02, -0.499], [-0.041, -0.4], [0.112, -0.416], [0.129, -0.533], [0.068, -0.404], [0.038, -0.314], [0.212, -0.326], [0.213, -0.406], [0.144, -0.311], [0.107, -0.232],
    ],
    "L_ines": [
        [-0.208, -0.091], [-0.39, -0.25], [-0.572, -0.338], [-0.715, -0.364], [-0.146, -0.51], [-0.06, -0.78], [0.003, -0.945], [0.056, -1.083], [-0.0, -0.47], [0.081, -0.565],
        [0.042, -0.405], [-0.014, -0.293], [0.121, -0.398], [0.182, -0.435], [0.118, -0.292], [0.051, -0.201], [0.215, -0.317], [0.257, -0.357], [0.197, -0.241], [0.131, -0.16],
    ],
    "O_pdf": [
        [0.123, -0.054], [0.227, -0.192], [0.335, -0.302], [0.413, -0.368], [0.004, -0.442], [0.173, -0.6], [0.314, -0.549], [0.381, -0.466], [0.015, -0.461], [0.205, -0.63],
        [0.342, -0.553], [0.384, -0.444], [0.046, -0.456], [0.217, -0.6], [0.339, -0.533], [0.377, -0.437], [0.084, -0.429], [0.208, -0.535], [0.295, -0.508], [0.34, -0.44],
    ],
    "O_ines": [
        [-0.179, -0.109], [-0.285, -0.263], [-0.361, -0.395], [-0.385, -0.506], [-0.08, -0.438], [-0.147, -0.633], [-0.265, -0.596], [-0.32, -0.51], [-0.031, -0.451], [-0.124, -0.656],
        [-0.251, -0.613], [-0.302, -0.528], [-0.008, -0.456], [-0.109, -0.645], [-0.233, -0.609], [-0.286, -0.532], [0.0, -0.447], [-0.093, -0.595], [-0.197, -0.589], [-0.249, -0.54],
    ],
    "S_pdf": [
        [-0.172, -0.154], [-0.262, -0.353], [-0.193, -0.502], [-0.043, -0.478], [-0.121, -0.486], [-0.103, -0.614], [-0.113, -0.509], [-0.133, -0.427], [-0.0, -0.46], [0.03, -0.579],
        [-0.009, -0.435], [-0.038, -0.355], [0.113, -0.412], [0.147, -0.535], [0.08, -0.389], [0.037, -0.299], [0.225, -0.345], [0.236, -0.451], [0.17, -0.357], [0.127, -0.279],
    ],
    "SIM_fim": [
        [-0.277, -0.024], [-0.515, -0.004], [-0.655, 0.086], [-0.719, 0.18], [-0.622, -0.236], [-0.712, 0.048], [-0.741, 0.253], [-0.761, 0.418], [-0.481, -0.245], [-0.588, 0.061],
        [-0.633, 0.275], [-0.656, 0.45], [-0.317, -0.215], [-0.432, 0.065], [-0.49, 0.273], [-0.52, 0.441], [-0.146, -0.155], [-0.255, 0.056], [-0.32, 0.226], [-0.365, 0.371],
    ],
    "U_pdf": [
        [-0.118, -0.127], [-0.138, -0.314], [-0.039, -0.448], [0.073, -0.511], [-0.101, -0.467], [-0.102, -0.661], [-0.087, -0.781], [-0.072, -0.882], [-0.0, -0.473], [0.012, -0.693],
        [0.014, -0.833], [0.025, -0.944], [0.089, -0.434], [0.125, -0.591], [0.065, -0.521], [0.026, -0.444], [0.17, -0.37], [0.167, -0.503], [0.11, -0.471], [0.066, -0.416],
    ],
    "U_ines": [
        [-0.133, -0.105], [-0.15, -0.263], [-0.057, -0.365], [0.061, -0.412], [-0.115, -0.487], [-0.089, -0.706], [-0.067, -0.842], [-0.045, -0.951], [0.0, -0.484], [0.033, -0.722],
        [0.056, -0.881], [0.075, -1.005], [0.103, -0.424], [0.154, -0.56], [0.115, -0.453], [0.074, -0.359], [0.191, -0.329], [0.226, -0.402], [0.182, -0.337], [0.134, -0.274],
    ],
    "V_pdf": [
        [-0.108, -0.12], [-0.13, -0.307], [-0.026, -0.443], [0.08, -0.515], [-0.113, -0.474], [-0.165, -0.661], [-0.185, -0.774], [-0.201, -0.866], [-0.0, -0.478], [0.049, -0.695],
        [0.068, -0.818], [0.09, -0.912], [0.091, -0.432], [0.112, -0.571], [0.058, -0.488], [0.034, -0.409], [0.157, -0.361], [0.154, -0.48], [0.11, -0.435], [0.082, -0.378],
    ],
    "V_ines": [
        [-0.129, -0.11], [-0.155, -0.249], [-0.072, -0.335], [0.037, -0.378], [-0.127, -0.491], [-0.184, -0.695], [-0.224, -0.824], [-0.248, -0.933], [-0.0, -0.483], [0.072, -0.716],
        [0.117, -0.865], [0.159, -0.981], [0.112, -0.42], [0.145, -0.49], [0.083, -0.391], [0.039, -0.319], [0.194, -0.322], [0.203, -0.376], [0.149, -0.309], [0.105, -0.258],
    ],
    "W_pdf": [
        [-0.125, -0.114], [-0.165, -0.285], [-0.039, -0.415], [0.087, -0.462], [-0.11, -0.446], [-0.151, -0.621], [-0.169, -0.728], [-0.187, -0.821], [-0.0, -0.467], [-0.007, -0.677],
        [-0.012, -0.802], [-0.012, -0.895], [0.09, -0.445], [0.134, -0.642], [0.141, -0.754], [0.155, -0.836], [0.166, -0.386], [0.158, -0.508], [0.113, -0.49], [0.089, -0.439],
    ],
    "W_ines": [
        [-0.114, -0.087], [-0.135, -0.206], [-0.039, -0.279], [0.072, -0.316], [-0.126, -0.473], [-0.138, -0.688], [-0.141, -0.814], [-0.135, -0.922], [0.0, -0.482], [0.057, -0.725],
        [0.101, -0.869], [0.134, -0.983], [0.102, -0.43], [0.188, -0.611], [0.232, -0.737], [0.269, -0.836], [0.181, -0.342], [0.193, -0.394], [0.16, -0.34], [0.138, -0.273],
    ],
    "Y_pdf": [
        [-0.185, -0.12], [-0.317, -0.297], [-0.367, -0.47], [-0.428, -0.591], [-0.117, -0.471], [-0.085, -0.602], [-0.114, -0.481], [-0.141, -0.37], [-0.0, -0.447], [0.026, -0.553],
        [-0.034, -0.407], [-0.073, -0.304], [0.109, -0.418], [0.142, -0.518], [0.057, -0.392], [-0.003, -0.295], [0.214, -0.379], [0.286, -0.546], [0.309, -0.644], [0.332, -0.746],
    ],
    "Y_ines": [
        [-0.182, -0.046], [-0.358, -0.168], [-0.522, -0.251], [-0.693, -0.294], [-0.15, -0.432], [-0.094, -0.254], [-0.055, -0.103], [-0.037, 0.027], [0.0, -0.463], [0.034, -0.246],
        [0.036, -0.088], [0.035, 0.049], [0.13, -0.431], [0.159, -0.276], [0.15, -0.129], [0.134, -0.002], [0.24, -0.357], [0.319, -0.441], [0.391, -0.523], [0.447, -0.587],
    ],
    "CINCO": [
        [-0.149, -0.152], [-0.22, -0.323], [-0.273, -0.446], [-0.35, -0.514], [-0.084, -0.495], [-0.068, -0.676], [-0.06, -0.768], [-0.056, -0.84], [0.0, -0.477], [0.041, -0.66],
        [0.057, -0.761], [0.07, -0.841], [0.075, -0.429], [0.126, -0.586], [0.147, -0.676], [0.163, -0.753], [0.145, -0.357], [0.224, -0.458], [0.267, -0.52], [0.306, -0.575],
    ],
    "OI_fim": [
        [-0.12, -0.103], [-0.149, -0.272], [-0.077, -0.407], [0.018, -0.469], [-0.113, -0.468], [-0.097, -0.627], [-0.108, -0.52], [-0.116, -0.413], [-0.0, -0.461], [0.039, -0.637],
        [0.004, -0.505], [-0.022, -0.393], [0.109, -0.424], [0.152, -0.588], [0.098, -0.484], [0.059, -0.393], [0.211, -0.364], [0.284, -0.501], [0.322, -0.589], [0.358, -0.68],
    ],
    "BOM_ini": [
        [-0.215, -0.135], [-0.366, -0.332], [-0.468, -0.483], [-0.562, -0.572], [-0.156, -0.427], [-0.143, -0.623], [-0.132, -0.76], [-0.123, -0.874], [0.0, -0.433], [0.06, -0.664],
        [0.077, -0.817], [0.091, -0.943], [0.116, -0.429], [0.172, -0.65], [0.186, -0.79], [0.189, -0.903], [0.195, -0.425], [0.228, -0.62], [0.234, -0.735], [0.228, -0.83],
    ],
    "BOM_fim": [
        [-0.293, -0.08], [-0.544, -0.17], [-0.763, -0.255], [-0.962, -0.292], [-0.21, -0.283], [-0.138, -0.432], [-0.14, -0.531], [-0.15, -0.659], [0.0, -0.374], [0.129, -0.587],
        [0.178, -0.734], [0.231, -0.899], [0.149, -0.465], [0.282, -0.718], [0.369, -0.901], [0.453, -1.086], [0.237, -0.534], [0.34, -0.774], [0.399, -0.939], [0.462, -1.104],
    ],
    "DIA_fim": [
        [-0.117, -0.174], [-0.126, -0.363], [-0.066, -0.515], [-0.02, -0.62], [-0.094, -0.467], [-0.095, -0.673], [-0.09, -0.796], [-0.079, -0.896], [-0.0, -0.464], [0.005, -0.612],
        [0.009, -0.696], [0.005, -0.77], [0.096, -0.432], [0.119, -0.593], [0.135, -0.702], [0.148, -0.799], [0.193, -0.369], [0.288, -0.469], [0.355, -0.526], [0.41, -0.579],
    ],
    "NAO_ind": [
        [-0.133, -0.124], [-0.202, -0.315], [-0.195, -0.467], [-0.12, -0.556], [-0.093, -0.503], [-0.11, -0.685], [-0.135, -0.809], [-0.154, -0.911], [-0.0, -0.475], [-0.021, -0.601],
        [-0.084, -0.512], [-0.115, -0.413], [0.089, -0.418], [0.054, -0.538], [-0.02, -0.449], [-0.051, -0.352], [0.18, -0.341], [0.13, -0.441], [0.062, -0.379], [0.031, -0.299],
    ],
    "EU_mao": [
        [0.095, -0.195], [0.264, -0.253], [0.39, -0.202], [0.455, -0.132], [0.416, -0.273], [0.458, -0.203], [0.397, -0.171], [0.34, -0.166], [0.439, -0.177], [0.466, -0.1],
        [0.394, -0.084], [0.335, -0.097], [0.427, -0.065], [0.44, -0.005], [0.367, -0.006], [0.314, -0.031], [0.396, 0.038], [0.386, 0.076], [0.322, 0.073], [0.28, 0.056],
    ],
    "VOCE_mao": [
        [-0.167, -0.077], [-0.32, -0.178], [-0.371, -0.254], [-0.297, -0.282], [-0.132, -0.49], [-0.147, -0.615], [-0.2, -0.639], [-0.241, -0.635], [-0.001, -0.483], [-0.128, -0.373],
        [-0.158, -0.262], [-0.133, -0.21], [0.104, -0.416], [-0.028, -0.296], [-0.055, -0.212], [-0.033, -0.185], [0.196, -0.327], [0.092, -0.244], [0.047, -0.177], [0.05, -0.153],
    ],
    "B_perfil": [
        [0.026, -0.188], [0.048, -0.378], [0.079, -0.516], [0.111, -0.618], [-0.021, -0.473], [0.019, -0.667], [0.066, -0.797], [0.101, -0.895], [0.0, -0.467], [0.031, -0.664],
        [0.073, -0.793], [0.114, -0.901], [0.049, -0.442], [0.077, -0.643], [0.117, -0.769], [0.157, -0.87], [0.122, -0.398], [0.142, -0.555], [0.161, -0.663], [0.18, -0.752],
    ],
    "OBRIGADO_mao": [
        [-0.089, -0.181], [-0.121, -0.38], [-0.155, -0.513], [-0.225, -0.581], [-0.059, -0.503], [0.017, -0.701], [0.063, -0.816], [0.091, -0.906], [0.0, -0.483], [0.078, -0.69],
        [0.12, -0.815], [0.146, -0.916], [0.065, -0.43], [0.133, -0.633], [0.165, -0.752], [0.18, -0.847], [0.135, -0.351], [0.179, -0.493], [0.201, -0.587], [0.219, -0.666],
    ],
}


def interpolar_molde(nome_a, nome_b, frac):
    molde_a = np.array([[0.0, 0.0]] + MOLDES_MAOS[nome_a], dtype=np.float64)
    if nome_a == nome_b:
        return molde_a
    molde_b = np.array([[0.0, 0.0]] + MOLDES_MAOS[nome_b], dtype=np.float64)
    return molde_a * (1.0 - frac) + molde_b * frac


def projetar_mao_quadro(
    pontos_molde,
    pulso_xy,
    tamanho_mao,
    rotacao_tela_graus,
    lado,
    escala_quadro_x,
    rng,
    escala_local_x=1.0,
    escala_local_y=1.0,
    ruido_pontos=0.007,
):
    m = pontos_molde.copy()
    m[:, 0] *= escala_local_x * lado
    m[:, 1] *= escala_local_y

    rad = math.radians(rotacao_tela_graus)
    c, s = math.cos(rad), math.sin(rad)
    rx = m[:, 0] * c - m[:, 1] * s
    ry = m[:, 0] * s + m[:, 1] * c

    pontos_3d = []
    for i in range(21):
        ruido_x = float(rng.normal(0.0, ruido_pontos)) if i > 0 else 0.0
        ruido_y = float(rng.normal(0.0, ruido_pontos)) if i > 0 else 0.0
        px = pulso_xy[0] + ((rx[i] + ruido_x) * tamanho_mao) / escala_quadro_x
        py = pulso_xy[1] + (ry[i] + ruido_y) * tamanho_mao
        pontos_3d.append([px, py, 0.0])
    return pontos_3d


def escolher_molde_par(rng, chave_a, chave_b, misturar=True):
    r = float(rng.random())
    if r < 0.42:
        return interpolar_molde(chave_a, chave_a, 0.0)
    if r < 0.84 or not misturar:
        return interpolar_molde(chave_b, chave_b, 0.0)
    peso = float(rng.uniform(0.2, 0.8))
    return interpolar_molde(chave_a, chave_b, peso)


def gerar_pose_referencia(nariz_x=0.50, nariz_y=0.34, larg_ombros=0.34):
    return [
        [nariz_x, nariz_y],
        [nariz_x - 0.04, nariz_y - 0.03],
        [nariz_x + 0.04, nariz_y - 0.03],
        [nariz_x - 0.09, nariz_y - 0.01],
        [nariz_x + 0.09, nariz_y - 0.01],
        [nariz_x - 0.03, nariz_y + 0.05],
        [nariz_x + 0.03, nariz_y + 0.05],
        [nariz_x - larg_ombros / 2.0, nariz_y + 0.28],
        [nariz_x + larg_ombros / 2.0, nariz_y + 0.28],
        [nariz_x - larg_ombros * 0.6, nariz_y + 0.50],
        [nariz_x + larg_ombros * 0.6, nariz_y + 0.50],
        [nariz_x - larg_ombros * 0.5, nariz_y + 0.65],
        [nariz_x + larg_ombros * 0.5, nariz_y + 0.65],
    ]


def configurar_trajetoria_sinal(sinal, rng, nariz_x, nariz_y, larg_ombros, lado):
    sinal_dir = 1.0 if lado > 0 else -1.0
    cfg = {
        "molde_ini": None,
        "molde_meio": None,
        "molde_fim": None,
        "rot_ini": 0.0,
        "rot_fim": 0.0,
        "pos_ini": (nariz_x - sinal_dir * 0.22, nariz_y + 0.22),
        "pos_fim": (nariz_x - sinal_dir * 0.22, nariz_y + 0.22),
        "osc_x": 0.0,
        "osc_y": 0.0,
        "osc_rot": 0.0,
        "osc_esc_x": 0.0,
        "estatico": False,
        "ancorado": False,
        "segunda_mao": None,
    }

    pares_letras = {
        "A": ("A_pdf", "A_ines", True),
        "B": ("B_pdf", "B_ines", True),
        "C": ("C_pdf", "C_ines", False),
        "D": ("D_pdf", "D_ines", False),
        "E": ("E_pdf", "E_ines", True),
        "F": ("F_pdf", "F_ines", False),
        "I": ("I_pdf", "I_ines", True),
        "L": ("L_pdf", "L_ines", True),
        "O": ("O_pdf", "O_ines", False),
        "U": ("U_pdf", "U_ines", True),
        "V": ("V_pdf", "V_ines", True),
        "W": ("W_pdf", "W_ines", True),
    }

    if sinal in pares_letras:
        ma, mb, misturar = pares_letras[sinal]
        molde = escolher_molde_par(rng, ma, mb, misturar=misturar)
        px = float(rng.uniform(0.18, 0.82))
        py = float(rng.uniform(0.36, 0.78))
        rot_max = 9.0 if sinal == "B" else (27.0 if sinal in ("C", "D", "A", "L") else 17.0)
        rot_base = float(rng.uniform(-14.0, rot_max)) * sinal_dir
        entrada_leve = bool(rng.random() < 0.32)
        dy_ent = float(rng.uniform(-0.050, -0.015)) if entrada_leve else float(rng.normal(0.0, 0.008))
        dx_ent = float(rng.normal(0.0, 0.018 if entrada_leve else 0.008))
        cfg.update(
            molde_ini=molde,
            molde_fim=molde,
            rot_ini=rot_base,
            rot_fim=rot_base + float(rng.uniform(-3.0, 3.0)),
            pos_ini=(px, py),
            pos_fim=(px + dx_ent, py + dy_ent),
            estatico=not entrada_leve,
        )
        return cfg

    if sinal == "Y":
        molde = escolher_molde_par(rng, "Y_pdf", "Y_ines", misturar=True)
        px = float(rng.uniform(0.20, 0.80))
        py = float(rng.uniform(0.38, 0.76))
        rot_base = float(rng.uniform(-14.0, 16.0))
        leve_descida = bool(rng.random() < 0.25)
        cfg.update(
            molde_ini=molde,
            molde_fim=molde,
            rot_ini=rot_base,
            rot_fim=rot_base + (6.0 * sinal_dir if leve_descida else float(rng.uniform(-2.0, 2.0))),
            pos_ini=(px, py),
            pos_fim=(px, py + (0.035 if leve_descida else float(rng.normal(0.0, 0.007)))),
            estatico=not leve_descida,
        )
        return cfg

    if sinal == "OI":
        variante = float(rng.random())
        px = nariz_x - sinal_dir * float(rng.uniform(0.14, 0.32))
        py = nariz_y + larg_ombros * float(rng.uniform(0.42, 0.78))
        if variante < 0.42:
            m_ini = escolher_molde_par(rng, "O_ines", "O_pdf", misturar=False)
            m_fim = escolher_molde_par(rng, "OI_fim", "I_ines", misturar=True)
            cfg.update(
                molde_ini=m_ini,
                molde_fim=m_fim,
                rot_ini=float(rng.uniform(16.0, 34.0)) * sinal_dir,
                rot_fim=float(rng.uniform(6.0, 18.0)) * sinal_dir,
                pos_ini=(px + sinal_dir * 0.05, py - 0.02),
                pos_fim=(px - sinal_dir * 0.04, py + 0.03),
                osc_x=float(rng.uniform(0.012, 0.030)),
                osc_rot=float(rng.uniform(8.0, 18.0)),
                osc_esc_x=0.14,
            )
        elif variante < 0.68:
            m_oi = escolher_molde_par(rng, "OI_fim", "I_ines", misturar=True)
            cfg.update(
                molde_ini=m_oi,
                molde_fim=m_oi,
                rot_ini=float(rng.uniform(4.0, 20.0)) * sinal_dir,
                rot_fim=float(rng.uniform(4.0, 20.0)) * sinal_dir,
                pos_ini=(px, py),
                pos_fim=(px - sinal_dir * 0.02, py),
                osc_x=float(rng.uniform(0.024, 0.055)),
                osc_rot=float(rng.uniform(14.0, 26.0)),
                osc_esc_x=0.18,
            )
        else:
            m_ola = interpolar_molde("CINCO", "CINCO", 0.0)
            py_ola = nariz_y + larg_ombros * float(rng.uniform(0.28, 0.62))
            px_ola = nariz_x - sinal_dir * float(rng.uniform(0.20, 0.36))
            so_punho = bool(rng.random() < 0.35)
            cfg.update(
                molde_ini=m_ola,
                molde_fim=m_ola,
                rot_ini=float(rng.uniform(-14.0, 8.0)) * sinal_dir,
                rot_fim=float(rng.uniform(-8.0, 10.0)) * sinal_dir,
                pos_ini=(px_ola, py_ola),
                pos_fim=(px_ola + sinal_dir * 0.02, py_ola + 0.02),
                osc_x=float(rng.uniform(0.018, 0.032) if so_punho else rng.uniform(0.040, 0.082)),
                osc_rot=float(rng.uniform(18.0, 28.0) if so_punho else rng.uniform(12.0, 22.0)),
            )
        return cfg

    if sinal == "BOM DIA":
        variante = float(rng.random())
        boca_x = nariz_x - sinal_dir * float(rng.uniform(0.06, 0.14))
        boca_y = nariz_y + larg_ombros * float(rng.uniform(0.76, 1.02))
        if variante < 0.58:
            m_ini = escolher_molde_par(rng, "O_ines", "BOM_ini", misturar=False)
            m_fim = interpolar_molde("BOM_fim", "CINCO", float(rng.uniform(0.0, 0.30)))
            cfg.update(
                molde_ini=m_ini,
                molde_fim=m_fim,
                rot_ini=float(rng.uniform(14.0, 28.0)) * sinal_dir,
                rot_fim=float(rng.uniform(22.0, 38.0)) * sinal_dir,
                pos_ini=(boca_x, boca_y),
                pos_fim=(boca_x - sinal_dir * float(rng.uniform(0.02, 0.08)), boca_y + float(rng.uniform(0.05, 0.12))),
                ancorado=True,
            )
        else:
            m_ini = escolher_molde_par(rng, "BOM_fim", "O_ines", misturar=False)
            m_meio = interpolar_molde("CINCO", "DIA_fim", 0.50)
            m_fim = interpolar_molde("DIA_fim", "DIA_fim", 0.0)
            cfg.update(
                molde_ini=m_ini,
                molde_meio=m_meio,
                molde_fim=m_fim,
                rot_ini=float(rng.uniform(14.0, 28.0)) * sinal_dir,
                rot_fim=float(rng.uniform(-6.0, 12.0)) * sinal_dir,
                pos_ini=(boca_x, boca_y),
                pos_fim=(nariz_x - sinal_dir * float(rng.uniform(0.24, 0.35)), nariz_y + larg_ombros * float(rng.uniform(0.62, 0.84))),
                ancorado=True,
            )
        return cfg

    if sinal == "OBRIGADO":
        usar_perfil = bool(rng.random() < 0.55)
        abre_leve = bool(rng.random() < 0.65)
        if usar_perfil:
            m_ini = interpolar_molde("OBRIGADO_mao", "B_ines", float(rng.uniform(0.0, 0.30)))
            m_fim = interpolar_molde("OBRIGADO_mao", "CINCO", float(rng.uniform(0.0, 0.35))) if abre_leve else m_ini
        else:
            m_ini = interpolar_molde("B_pdf", "B_ines", 0.35) if abre_leve else escolher_molde_par(rng, "B_ines", "B_pdf", misturar=True)
            m_fim = interpolar_molde("B_ines", "CINCO", float(rng.uniform(0.20, 0.50))) if abre_leve else m_ini
        ini_x = nariz_x - sinal_dir * larg_ombros * float(rng.uniform(0.35, 0.65))
        ini_y = nariz_y + larg_ombros * float(rng.uniform(0.02, 0.22))
        afastar_lateral = bool(rng.random() < 0.65)
        fim_x = ini_x - sinal_dir * larg_ombros * float(rng.uniform(0.18, 0.36))
        fim_y = ini_y + (float(rng.uniform(0.02, 0.07)) if afastar_lateral else float(rng.uniform(0.10, 0.22)))
        cfg.update(
            molde_ini=m_ini,
            molde_fim=m_fim,
            rot_ini=float(rng.uniform(14.0, 34.0)) * sinal_dir,
            rot_fim=float(rng.uniform(10.0, 26.0)) * sinal_dir,
            pos_ini=(ini_x, ini_y),
            pos_fim=(fim_x, fim_y),
            osc_x=float(rng.uniform(0.012, 0.032)) if afastar_lateral else 0.0,
            osc_y=float(rng.uniform(0.010, 0.026)) if afastar_lateral else 0.0,
            ancorado=True,
        )
        return cfg

    if sinal == "POR FAVOR":
        m_b = interpolar_molde("B_perfil", "B_ines", float(rng.uniform(0.0, 0.40))) if rng.random() < 0.50 else escolher_molde_par(rng, "B_ines", "B_pdf", misturar=True)
        peito_y = nariz_y + larg_ombros * float(rng.uniform(0.72, 1.04))
        usar_duas = bool(rng.random() < 0.70)
        ini_x = nariz_x - sinal_dir * float(rng.uniform(0.02, 0.12))
        fim_x = ini_x - sinal_dir * float(rng.uniform(0.05, 0.12))
        cfg.update(
            molde_ini=m_b,
            molde_fim=m_b,
            rot_ini=float(rng.uniform(46.0, 68.0)) * sinal_dir,
            rot_fim=float(rng.uniform(40.0, 62.0)) * sinal_dir,
            pos_ini=(ini_x, peito_y),
            pos_fim=(fim_x, peito_y + float(rng.uniform(-0.02, 0.04))),
            osc_x=0.015 if usar_duas else float(rng.uniform(0.030, 0.060)),
            ancorado=True,
        )
        if usar_duas:
            base_x = nariz_x + sinal_dir * float(rng.uniform(0.16, 0.26))
            base_y = peito_y + float(rng.uniform(0.01, 0.05))
            cfg["segunda_mao"] = {
                "molde": m_b,
                "pos_ini": (base_x, base_y),
                "pos_fim": (base_x - sinal_dir * 0.04, base_y),
                "rot_ini": float(rng.uniform(-42.0, -22.0)) * sinal_dir,
                "rot_fim": float(rng.uniform(-38.0, -20.0)) * sinal_dir,
                "lado": -lado,
            }
        return cfg

    if sinal == "SIM":
        px = nariz_x + sinal_dir * float(rng.uniform(0.06, 0.26))
        py = nariz_y + larg_ombros * float(rng.uniform(0.40, 1.15))
        nod_frontal = bool(rng.random() < 0.60)
        if nod_frontal:
            cfg.update(
                molde_ini=interpolar_molde("S_pdf", "S_pdf", 0.0),
                molde_fim=interpolar_molde("SIM_fim", "SIM_fim", 0.0),
                rot_ini=float(rng.uniform(-16.0, 8.0)) * sinal_dir,
                rot_fim=float(rng.uniform(-54.0, -28.0)) * sinal_dir,
                pos_ini=(px, py),
                pos_fim=(px + sinal_dir * float(rng.uniform(0.02, 0.06)), py + float(rng.uniform(0.05, 0.11))),
                osc_y=float(rng.uniform(0.015, 0.035)),
                ancorado=True,
            )
        else:
            m_s = interpolar_molde("S_pdf", "SIM_fim", float(rng.uniform(0.0, 0.45)))
            cfg.update(
                molde_ini=m_s,
                molde_fim=m_s,
                rot_ini=float(rng.uniform(-20.0, 10.0)) * sinal_dir,
                rot_fim=float(rng.uniform(-34.0, -6.0)) * sinal_dir,
                pos_ini=(px, py),
                pos_fim=(px, py + float(rng.uniform(0.03, 0.08))),
                osc_y=float(rng.uniform(0.038, 0.070)),
                osc_rot=float(rng.uniform(12.0, 22.0)),
                ancorado=True,
            )
        return cfg

    if sinal == "NAO":
        usar_y = bool(rng.random() < 0.42)
        px = nariz_x - sinal_dir * float(rng.uniform(0.14, 0.34))
        py = nariz_y + larg_ombros * float(rng.uniform(0.55, 0.98))
        so_punho = bool(rng.random() < 0.40)
        if usar_y:
            molde = escolher_molde_par(rng, "Y_ines", "Y_pdf", misturar=True)
            cfg.update(
                molde_ini=molde,
                molde_fim=molde,
                rot_ini=float(rng.uniform(-14.0, 16.0)),
                rot_fim=float(rng.uniform(-14.0, 16.0)),
                pos_ini=(px, py),
                pos_fim=(px + sinal_dir * float(rng.uniform(-0.04, 0.04)), py + float(rng.uniform(0.0, 0.08))),
                osc_x=float(rng.uniform(0.024, 0.042) if so_punho else rng.uniform(0.042, 0.078)),
                osc_rot=float(rng.uniform(18.0, 30.0)),
            )
        else:
            molde = interpolar_molde("NAO_ind", "NAO_ind", 0.0)
            cfg.update(
                molde_ini=molde,
                molde_fim=molde,
                rot_ini=float(rng.uniform(-12.0, 12.0)),
                rot_fim=float(rng.uniform(-10.0, 12.0)),
                pos_ini=(px, py),
                pos_fim=(px + sinal_dir * float(rng.uniform(-0.03, 0.03)), py + float(rng.uniform(0.0, 0.04))),
                osc_x=float(rng.uniform(0.018, 0.034) if so_punho else rng.uniform(0.040, 0.078)),
                osc_rot=float(rng.uniform(16.0, 28.0)),
            )
        return cfg

    if sinal == "EU":
        usar_molde_eu = bool(rng.random() < 0.65)
        peito_x = nariz_x + sinal_dir * float(rng.uniform(0.06, 0.15))
        peito_y = nariz_y + larg_ombros * float(rng.uniform(0.76, 1.00))
        if usar_molde_eu:
            molde = interpolar_molde("EU_mao", "EU_mao", 0.0)
            rot_i = float(rng.uniform(-12.0, 12.0)) * sinal_dir
        else:
            molde = interpolar_molde("NAO_ind", "NAO_ind", 0.0)
            rot_i = float(rng.uniform(-88.0, -62.0)) * sinal_dir
        cfg.update(
            molde_ini=molde,
            molde_fim=molde,
            rot_ini=rot_i,
            rot_fim=rot_i + float(rng.uniform(-5.0, 5.0)),
            pos_ini=(peito_x + sinal_dir * 0.04, peito_y),
            pos_fim=(peito_x, peito_y + float(rng.uniform(-0.01, 0.02))),
            ancorado=True,
        )
        return cfg

    if sinal == "VOCE":
        molde = interpolar_molde("VOCE_mao", "NAO_ind", float(rng.uniform(0.0, 0.28)))
        px = nariz_x + sinal_dir * float(rng.uniform(0.03, 0.14))
        py = nariz_y + larg_ombros * float(rng.uniform(0.42, 0.85))
        cfg.update(
            molde_ini=molde,
            molde_fim=molde,
            rot_ini=float(rng.uniform(-8.0, 12.0)) * sinal_dir,
            rot_fim=float(rng.uniform(-6.0, 14.0)) * sinal_dir,
            pos_ini=(px, py),
            pos_fim=(px + sinal_dir * float(rng.uniform(-0.01, 0.03)), py + float(rng.uniform(0.02, 0.09))),
            ancorado=True,
        )
        return cfg

    if sinal == "CASA":
        m_b = interpolar_molde("B_perfil", "B_ines", float(rng.uniform(0.0, 0.35))) if rng.random() < 0.65 else escolher_molde_par(rng, "B_ines", "B_pdf", misturar=True)
        usar_duas = bool(rng.random() < 0.84)
        altura_y = nariz_y + larg_ombros * float(rng.uniform(0.62, 0.90))
        meia_dist = float(rng.uniform(0.10, 0.18))
        toque_duplo = bool(usar_duas and rng.random() < 0.70)
        inclinacao = float(rng.uniform(10.0, 42.0))
        cfg.update(
            molde_ini=m_b,
            molde_fim=m_b,
            rot_ini=inclinacao * sinal_dir,
            rot_fim=(inclinacao + float(rng.uniform(-4.0, 6.0))) * sinal_dir,
            pos_ini=(nariz_x - sinal_dir * (meia_dist + 0.03), altura_y + 0.02),
            pos_fim=(nariz_x - sinal_dir * meia_dist, altura_y - (0.01 if usar_duas else 0.08)),
            osc_x=float(rng.uniform(0.025, 0.065)) if toque_duplo else 0.0,
            osc_y=float(rng.uniform(0.015, 0.045)) if toque_duplo else 0.0,
            ancorado=True,
        )
        if usar_duas:
            inclinacao2 = inclinacao + float(rng.uniform(-6.0, 6.0))
            cfg["segunda_mao"] = {
                "molde": m_b,
                "pos_ini": (nariz_x + sinal_dir * (meia_dist + 0.03), altura_y + float(rng.uniform(0.0, 0.03))),
                "pos_fim": (nariz_x + sinal_dir * meia_dist, altura_y + float(rng.uniform(-0.02, 0.02))),
                "rot_ini": -inclinacao2 * sinal_dir,
                "rot_fim": -(inclinacao2 + float(rng.uniform(-4.0, 4.0))) * sinal_dir,
                "lado": -lado,
            }
        return cfg

    if sinal == "AJUDA":
        variante = float(rng.random())
        peito_y = nariz_y + larg_ombros * float(rng.uniform(0.60, 0.98))
        m_b = escolher_molde_par(rng, "B_ines", "B_pdf", misturar=True)
        if variante < 0.40:
            px = nariz_x + sinal_dir * float(rng.uniform(0.02, 0.14))
            desl_x = sinal_dir * float(rng.uniform(0.05, 0.14))
            cfg.update(
                molde_ini=m_b,
                molde_fim=m_b,
                rot_ini=float(rng.uniform(-2.0, 14.0)) * sinal_dir,
                rot_fim=float(rng.uniform(2.0, 16.0)) * sinal_dir,
                pos_ini=(px, peito_y),
                pos_fim=(px + desl_x, peito_y + float(rng.uniform(-0.02, 0.05))),
                osc_x=float(rng.uniform(0.015, 0.045)),
                ancorado=True,
                segunda_mao={
                    "molde": m_b,
                    "pos_ini": (px + sinal_dir * float(rng.uniform(0.07, 0.13)), peito_y + float(rng.uniform(0.05, 0.10))),
                    "pos_fim": (px + desl_x + sinal_dir * float(rng.uniform(0.07, 0.13)), peito_y + float(rng.uniform(0.05, 0.11))),
                    "rot_ini": float(rng.uniform(-62.0, -35.0)) * sinal_dir,
                    "rot_fim": float(rng.uniform(-58.0, -30.0)) * sinal_dir,
                    "lado": -lado,
                },
            )
        elif variante < 0.80:
            m_a = escolher_molde_par(rng, "A_ines", "A_pdf", misturar=True)
            px = nariz_x - sinal_dir * float(rng.uniform(0.02, 0.08))
            py_a = nariz_y + larg_ombros * float(rng.uniform(0.52, 0.72))
            desl_x = -sinal_dir * float(rng.uniform(0.05, 0.15))
            cfg.update(
                molde_ini=m_a,
                molde_fim=m_a,
                rot_ini=float(rng.uniform(8.0, 22.0)) * sinal_dir,
                rot_fim=float(rng.uniform(10.0, 24.0)) * sinal_dir,
                pos_ini=(px, py_a + 0.02),
                pos_fim=(px + desl_x, py_a - 0.01),
                osc_x=float(rng.uniform(0.015, 0.040)),
                ancorado=True,
                segunda_mao={
                    "molde": m_a if rng.random() < 0.5 else m_b,
                    "pos_ini": (px + sinal_dir * float(rng.uniform(0.07, 0.12)), py_a + float(rng.uniform(0.05, 0.09))),
                    "pos_fim": (px + desl_x + sinal_dir * float(rng.uniform(0.09, 0.16)), py_a + float(rng.uniform(0.04, 0.09))),
                    "rot_ini": float(rng.uniform(-18.0, -4.0)) * sinal_dir,
                    "rot_fim": float(rng.uniform(-16.0, -2.0)) * sinal_dir,
                    "lado": -lado,
                },
            )
        else:
            px = nariz_x + sinal_dir * float(rng.uniform(0.05, 0.18))
            cfg.update(
                molde_ini=m_b,
                molde_fim=m_b,
                rot_ini=float(rng.uniform(0.0, 14.0)) * sinal_dir,
                rot_fim=float(rng.uniform(0.0, 14.0)) * sinal_dir,
                pos_ini=(px, peito_y + 0.07),
                pos_fim=(px, peito_y - 0.09),
                ancorado=True,
            )
        return cfg

    return cfg


def gerar_amostra_temporal(sinal, rng, lado=1.0, incluir_pose=True):
    largura_quadro = 480
    altura_quadro = 640
    escala_quadro_x = largura_quadro / altura_quadro

    nariz_x = float(rng.uniform(0.38, 0.62))
    nariz_y = float(rng.uniform(0.26, 0.42) if incluir_pose else rng.uniform(0.34, 0.46))
    larg_ombros = float(rng.uniform(0.25, 0.42) if incluir_pose else rng.uniform(0.32, 0.44))
    pose = gerar_pose_referencia(nariz_x, nariz_y, larg_ombros) if incluir_pose else None

    cfg = configurar_trajetoria_sinal(sinal, rng, nariz_x, nariz_y, larg_ombros, lado)
    tamanho_mao = float(rng.uniform(0.18, 0.36))
    esc_local_x = float(rng.uniform(0.88, 1.14))
    esc_local_y = float(rng.uniform(0.90, 1.12))
    inclinacao_extra = float(rng.uniform(-5.0, 5.0) if cfg["ancorado"] else rng.uniform(-10.0, 10.0))

    estatico = bool(cfg["estatico"])
    total_quadros = int(rng.integers(3, 9))
    fase_ini = float(rng.uniform(0.0, math.pi * 2.0))
    ciclos = float(rng.uniform(0.65, 1.55))
    prog_ini = 0.0 if estatico else float(rng.uniform(0.0, 0.18))
    prog_fim = 1.0 if estatico else float(rng.uniform(0.76, 1.0))

    inverter_foco_duplo = bool(
        cfg["segunda_mao"] is not None
        and sinal in ("CASA", "POR FAVOR")
        and rng.random() < 0.45
    )

    quadros = []
    for t in range(total_quadros):
        frac = t / max(total_quadros - 1, 1)
        prog = prog_ini + (prog_fim - prog_ini) * frac

        if cfg["molde_meio"] is not None:
            if prog < 0.5:
                p_loc = prog * 2.0
                molde_t = cfg["molde_ini"] * (1.0 - p_loc) + cfg["molde_meio"] * p_loc
            else:
                p_loc = (prog - 0.5) * 2.0
                molde_t = cfg["molde_meio"] * (1.0 - p_loc) + cfg["molde_fim"] * p_loc
        else:
            molde_t = cfg["molde_ini"] * (1.0 - prog) + cfg["molde_fim"] * prog

        ang_onda = fase_ini + frac * math.pi * 2.0 * ciclos
        onda_x = math.sin(ang_onda) * cfg["osc_x"]
        onda_y = math.sin(ang_onda) * cfg["osc_y"]
        onda_rot = math.sin(ang_onda) * cfg["osc_rot"]
        esc_x_t = esc_local_x * (1.0 - cfg["osc_esc_x"] * abs(math.sin(ang_onda)))

        rot_base = cfg["rot_ini"] + (cfg["rot_fim"] - cfg["rot_ini"]) * prog
        rot_final = rot_base + inclinacao_extra + onda_rot

        px = cfg["pos_ini"][0] + (cfg["pos_fim"][0] - cfg["pos_ini"][0]) * prog + onda_x + float(rng.normal(0.0, 0.003))
        py = cfg["pos_ini"][1] + (cfg["pos_fim"][1] - cfg["pos_ini"][1]) * prog + onda_y + float(rng.normal(0.0, 0.003))

        mao1_3d = projetar_mao_quadro(
            molde_t,
            (px, py),
            tamanho_mao,
            rot_final,
            lado,
            escala_quadro_x,
            rng,
            escala_local_x=esc_x_t,
            escala_local_y=esc_local_y,
        )

        mao2_3d = None
        if cfg["segunda_mao"] is not None:
            sm = cfg["segunda_mao"]
            rot2_b = sm["rot_ini"] + (sm["rot_fim"] - sm["rot_ini"]) * prog
            rot2_f = rot2_b + inclinacao_extra * 0.5 - onda_rot * 0.5
            px2 = sm["pos_ini"][0] + (sm["pos_fim"][0] - sm["pos_ini"][0]) * prog - onda_x * 0.6 + float(rng.normal(0.0, 0.003))
            py2 = sm["pos_ini"][1] + (sm["pos_fim"][1] - sm["pos_ini"][1]) * prog + onda_y * 0.6 + float(rng.normal(0.0, 0.003))
            mao2_3d = projetar_mao_quadro(
                sm["molde"],
                (px2, py2),
                tamanho_mao * float(rng.uniform(0.95, 1.05)),
                rot2_f,
                sm["lado"],
                escala_quadro_x,
                rng,
                escala_local_x=esc_local_x,
                escala_local_y=esc_local_y,
            )

        if inverter_foco_duplo and mao2_3d is not None:
            quadro = calcular_vetor_quadro(mao2_3d, largura_quadro, altura_quadro, pose, mao1_3d)
        else:
            quadro = calcular_vetor_quadro(mao1_3d, largura_quadro, altura_quadro, pose, mao2_3d)

        if quadro is not None:
            quadros.append(quadro)

    if estatico and rng.random() < 0.35:
        return montar_vetor_temporal([quadros[-1]])
    return montar_vetor_temporal(quadros)


def criar_dataset():
    sinais = [
        "OI", "BOM DIA", "OBRIGADO", "POR FAVOR",
        "SIM", "NAO", "EU", "VOCE", "CASA", "AJUDA",
        "A", "B", "C", "D", "E", "F", "I", "L", "O", "U", "V", "W", "Y",
    ]
    rng = np.random.default_rng(42)
    amostras_por_configuracao = 80
    caminho_csv = os.path.join(os.path.dirname(__file__), "dataset_libras.csv")

    cabecalho = ["rotulo"] + [f"coord_{i}" for i in range(110)]
    linhas = [cabecalho]

    for sinal in sinais:
        qtd = 110 if sinal in ("OI", "BOM DIA", "AJUDA", "NAO", "CASA", "POR FAVOR") else amostras_por_configuracao
        ancorado_rosto = sinal in ("OBRIGADO", "BOM DIA", "EU", "VOCE")
        ancorado_peito = sinal in ("CASA", "POR FAVOR", "AJUDA")
        for lado in (1.0, -1.0):
            for idx_amostra in range(qtd * 2):
                if ancorado_rosto:
                    incluir_pose = idx_amostra % 6 != 0
                elif ancorado_peito:
                    incluir_pose = idx_amostra % 4 != 0
                else:
                    incluir_pose = idx_amostra % 2 == 0
                vetor = gerar_amostra_temporal(sinal, rng, lado=lado, incluir_pose=incluir_pose)
                linhas.append([sinal] + [round(float(v), 6) for v in vetor])

    with open(caminho_csv, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerows(linhas)

    print(f"Dataset criado com {len(linhas) - 1} amostras ({len(sinais)} sinais, 110 caracteristicas) em {caminho_csv}")


if __name__ == "__main__":
    criar_dataset()
